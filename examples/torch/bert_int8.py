"""
BERT INT8 Quantization Test

This module mirrors the flow of `vit_int8.py` but targets BERT tiny-size
sequence-classification models from HuggingFace.

It provides two test paths:
1) first encoder layer quantization test
2) full sequence-classification model quantization test

You can keep `model_path` as a HuggingFace repository id or replace it with
your local model directory later.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

import allo
import numpy as np
import torch
import torch.nn as nn

from allo.ops.bert import (
    BertGetFirstToken,
    BertLayer,
    BertForSequenceClassification,
)

from allo.quant.quant_config import (
    Calibrator,
    get_vit_optimized_config,
    replace_module_with_quantized,
    set_extra_compile_param_for_config,
)
from allo.quant.quant_modules import (
    IntGELU,
    IntLayerNorm,
    IntSoftmax,
    IntSoftmaxWithMask,
    QAdd,
    QLinear,
    QMatMul,
    QMatMulIsqrtD,
)


# ==============================================================================
# BERT Tiny Configurations
# ==============================================================================

BERT_CONFIGS = {
    "bert-tiny-cls": {
        # Placeholder model repo/path for tiny sequence classification.
        # Replace it with your local path when ready.
        # Example local path: "/root/data/models/bert-tiny-sst2"
        "model_path": "/root/data/models/bert-tiny-finetuned-mnli",  # "your-org/bert-tiny-seq-cls",
        "max_length": 128,
    },
    "bert-tiny-qnli": {
        "model_path": "/root/data/models/bert-tiny-finetuned-qnli",
        "max_length": 128,
    },
}

MNLI_SPLIT_TO_FILE = {
    "train": "multinli_1.0_train.jsonl",
    "dev_matched": "multinli_1.0_dev_matched.jsonl",
    "dev_mismatched": "multinli_1.0_dev_mismatched.jsonl",
}

DEFAULT_MNLI_ROOT = "/root/data/dataset/multinli_1.0/"

QNLI_SPLIT_TO_FILE = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
}

DEFAULT_QNLI_ROOT = "/root/data/dataset/QNLI/"

FIRST_LAYER_MUTABLE_MODULES = [
    "attention.linear_q",
    "attention.linear_k",
    "attention.linear_v",
    "attention.matmul1",
    "attention.softmax",
    "attention.matmul2",
    "attention.linear_out",
    "add1",
    "norm1",
    "ffn.fc1",
    "ffn.activation",
    "ffn.fc2",
    "add2",
    "norm2",
]


SAMPLE_TEXT_POOL = [
    "This movie is fantastic and surprisingly thoughtful.",
    "The product quality is poor and I want a refund.",
    "Service was okay, nothing special but acceptable.",
    "Absolutely loved the experience, will come back again.",
    "The plot is boring and the acting feels flat.",
    "Great value for money and fast delivery.",
    "I do not recommend this to anyone.",
    "Performance is decent for the price point.",
    "User interface is clean and easy to navigate.",
    "Battery life is short and charging is very slow.",
]

CALIBRATION_FULL_LENGTH_SENTENCE = (
    "This is a deliberately long calibration sentence containing diverse words about products, services, "
    "quality, delivery, reliability, customer experience, usability, performance, battery, interface, "
    "support, expectations, and outcomes. "
    * 24
)


def _build_text_samples(num_samples: int) -> List[str]:
    if num_samples <= 0:
        return []
    pool_size = len(SAMPLE_TEXT_POOL)
    return [SAMPLE_TEXT_POOL[i % pool_size] for i in range(num_samples)]


def _build_calibration_text_samples(num_samples: int, max_length: int) -> List[str]:
    if num_samples <= 0:
        return []

    short_target = max(4, max_length // 6)
    medium_target = max(short_target + 2, max_length // 2)
    long_target = max(medium_target + 2, int(max_length * 5 / 6))

    def _build_target_len_sentence(target_words: int, seed_idx: int) -> str:
        words = []
        base = SAMPLE_TEXT_POOL[seed_idx % len(SAMPLE_TEXT_POOL)].lower().replace(".", "").split()
        while len(words) < target_words:
            words.extend(base)
        words = words[:target_words]
        return " ".join(words) + "."

    short_count = num_samples // 3
    medium_count = num_samples // 3
    long_count = num_samples - short_count - medium_count

    samples: List[str] = []
    for i in range(short_count):
        samples.append(_build_target_len_sentence(short_target, i))
    for i in range(medium_count):
        samples.append(_build_target_len_sentence(medium_target, i + short_count))
    for i in range(long_count):
        samples.append(_build_target_len_sentence(long_target, i + short_count + medium_count))

    if len(samples) > 0:
        samples[-1] = CALIBRATION_FULL_LENGTH_SENTENCE
    return samples


def _load_hf_model_and_tokenizer(model_path: str):
    from transformers import AutoTokenizer, BertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).eval()
    return model, tokenizer


def _tokenize_batch(
    tokenizer,
    texts: List[str],
    max_length: int,
    pair_texts: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    if pair_texts is None:
        batch = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        batch = tokenizer(
            texts,
            pair_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    if "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])  # BERT-compatible fallback
    return batch


def _build_label_to_id(hf_model: nn.Module, task: str) -> Dict[str, int]:
    if task == "mnli":
        canonical = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }
    elif task == "qnli":
        canonical = {
            "entailment": 0,
            "not_entailment": 1,
            "not entailment": 1,
            "0": 0,
            "1": 1,
        }
    else:
        raise ValueError(f"Unsupported task for label mapping: {task}")

    cfg_label2id = getattr(hf_model.config, "label2id", None)
    if isinstance(cfg_label2id, dict) and len(cfg_label2id) > 0:
        normalized = {}
        for k, v in cfg_label2id.items():
            if isinstance(k, str):
                normalized[k.strip().lower()] = int(v)

        if len(normalized) > 0:
            # Preferred path: config already contains semantic MNLI labels.
            if task == "mnli":
                if any(name in normalized for name in ("entailment", "neutral", "contradiction")):
                    semantic_map = {
                        name: normalized[name]
                        for name in ("entailment", "neutral", "contradiction")
                        if name in normalized
                    }
                    if len(semantic_map) == 3:
                        return semantic_map
            elif task == "qnli":
                if "entailment" in normalized and "not_entailment" in normalized:
                    semantic_map = {
                        "entailment": normalized["entailment"],
                        "not_entailment": normalized["not_entailment"],
                        "not entailment": normalized["not_entailment"],
                        "0": normalized["entailment"],
                        "1": normalized["not_entailment"],
                    }
                    return semantic_map

            # Generic path: many checkpoints store LABEL_0/LABEL_1/LABEL_2 in config.
            # In this case, keep a stable MNLI default mapping so gold_label can be parsed.
            generic_keys = {f"label_{i}" for i in range(len(normalized))}
            if task == "mnli" and set(normalized.keys()) == generic_keys and len(normalized) == 3:
                return canonical
            if task == "qnli" and set(normalized.keys()) == generic_keys and len(normalized) == 2:
                return canonical

            # If config is non-empty but unusable for MNLI string labels, fall back safely.
            return canonical

    return canonical


def _load_mnli_split(
    mnli_root: str,
    split: str,
    limit: int,
    label_to_id: Dict[str, int],
) -> Tuple[List[str], List[str], List[int]]:
    if split not in MNLI_SPLIT_TO_FILE:
        raise ValueError(f"Unknown MNLI split: {split}. Available: {list(MNLI_SPLIT_TO_FILE.keys())}")

    file_path = os.path.join(mnli_root, MNLI_SPLIT_TO_FILE[split])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MNLI file not found: {file_path}")

    premises: List[str] = []
    hypotheses: List[str] = []
    labels: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and len(labels) >= limit:
                break
            sample = json.loads(line)
            raw_label = str(sample.get("gold_label", "")).strip().lower()
            if raw_label not in label_to_id:
                continue

            s1 = sample.get("sentence1", None)
            s2 = sample.get("sentence2", None)
            if not isinstance(s1, str) or not isinstance(s2, str):
                continue

            premises.append(s1)
            hypotheses.append(s2)
            labels.append(label_to_id[raw_label])

    if len(labels) == 0:
        raise RuntimeError(
            f"No valid MNLI samples loaded from {file_path}. "
            "Please check dataset content and label mapping."
        )

    return premises, hypotheses, labels


def _load_qnli_split(
    qnli_root: str,
    split: str,
    limit: int,
    label_to_id: Dict[str, int],
) -> Tuple[List[str], List[str], List[int]]:
    if split not in QNLI_SPLIT_TO_FILE:
        raise ValueError(f"Unknown QNLI split: {split}. Available: {list(QNLI_SPLIT_TO_FILE.keys())}")

    file_path = os.path.join(qnli_root, QNLI_SPLIT_TO_FILE[split])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"QNLI file not found: {file_path}")

    questions: List[str] = []
    sentences: List[str] = []
    labels: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if limit > 0 and len(labels) >= limit:
                break

            q = row.get("question", None)
            s = row.get("sentence", None)
            if not isinstance(q, str) or not isinstance(s, str):
                continue

            raw_label = row.get("label", None)
            if not isinstance(raw_label, str):
                continue

            label_key = raw_label.strip().lower()
            if label_key not in label_to_id:
                continue

            questions.append(q)
            sentences.append(s)
            labels.append(label_to_id[label_key])

    if len(labels) == 0:
        raise RuntimeError(
            f"No valid QNLI samples loaded from {file_path}. "
            "Please check TSV format and label mapping. "
            "Note: test split has no labels and cannot be used for accuracy evaluation."
        )

    return questions, sentences, labels


def _prepare_first_layer_inputs(
    model: nn.Module,
    token_batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        hidden_states = model.bert.embeddings(
            input_ids=token_batch["input_ids"],
            token_type_ids=token_batch.get("token_type_ids", None),
        )
        attention_bias = model.bert.build_attention_bias(
            token_batch["attention_mask"],
            hidden_states.dtype,
        )

    return hidden_states, attention_bias


def _print_diff_stats(name: str, golden: torch.Tensor, quantized: torch.Tensor) -> None:
    mean_diff = torch.mean(torch.abs(golden - quantized)).item()
    max_diff = torch.max(torch.abs(golden - quantized)).item()
    print("\n" + "-" * 40)
    print(f"{name} Results (float vs fakequant):")
    print(f"    Mean diff: {mean_diff:.6f}")
    print(f"    Max diff: {max_diff:.6f}")


def compare_tensors_with_plot(
    golden,
    ref,
    name: str = "Tensor Comparison",
    tolerance_rtol: float = 1e-5,
    tolerance_atol: float = 1e-6,
    enable_plot: bool = True,
    plot_filename: str = "tensor_comparison.png",
):
    if isinstance(golden, torch.Tensor):
        golden = golden.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()

    golden_flat = golden.flatten()
    ref_flat = ref.flatten()

    if len(golden_flat) != len(ref_flat):
        print(f"Error: Size mismatch - {golden_flat.shape} vs {ref_flat.shape}")
        return {"matched": False, "error": "Size mismatch"}

    diff = np.abs(golden_flat - ref_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    max_rel_diff = np.max(diff / (np.abs(ref_flat) + 1e-8))

    print("\n" + "=" * 80)
    print(f"{name} - Tensor Comparison")
    print("=" * 80)
    print(f"Golden shape:  {golden.shape}")
    print(f"Ref shape:     {ref.shape}")
    print(f"Golden stats:  mean={golden_flat.mean():.6f}, std={golden_flat.std():.6f}")
    print(f"               min={golden_flat.min():.6f}, max={golden_flat.max():.6f}")
    print(f"Ref stats:     mean={ref_flat.mean():.6f}, std={ref_flat.std():.6f}")
    print(f"               min={ref_flat.min():.6f}, max={ref_flat.max():.6f}")
    print(f"  abs diff - mean: {mean_diff:.6f}, max: {max_diff:.6f}")
    print(f"  rel diff - max: {max_rel_diff:.6f}")

    plot_path = None
    if enable_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            indices = np.arange(len(golden_flat))
            sample_step = max(1, len(golden_flat) // 1000)
            sample_indices = indices[::sample_step]

            axes[0, 0].plot(sample_indices, golden_flat[::sample_step], label="Golden", alpha=0.7, linewidth=0.5)
            axes[0, 0].plot(sample_indices, ref_flat[::sample_step], label="Ref (Quantized)", alpha=0.7, linewidth=0.5)
            axes[0, 0].set_title("Full Output Comparison (Sampled)")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(sample_indices, diff[::sample_step], color="red", alpha=0.7, linewidth=0.5)
            axes[0, 1].set_title("Absolute Difference")
            axes[0, 1].grid(True, alpha=0.3)

            num_points = min(100, len(golden_flat))
            axes[1, 0].plot(np.arange(num_points), golden_flat[:num_points], label="Golden", marker="o", markersize=3, alpha=0.7)
            axes[1, 0].plot(np.arange(num_points), ref_flat[:num_points], label="Ref", marker="s", markersize=3, alpha=0.7)
            axes[1, 0].set_title("First 100 Points")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].scatter(golden_flat[::sample_step], ref_flat[::sample_step], alpha=0.5, s=1)
            axes[1, 1].plot([golden_flat.min(), golden_flat.max()], [golden_flat.min(), golden_flat.max()], "r--", label="y=x", linewidth=1)
            axes[1, 1].set_title("Scatter: Golden vs Ref")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"{plot_filename[:-4]}_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"✓ Comparison plot saved to: {plot_path}")
            plt.close()
        except ImportError:
            print("Warning: matplotlib not available, skip visualization")

    matched = np.allclose(golden_flat, ref_flat, rtol=tolerance_rtol, atol=tolerance_atol)
    print(f"Matched: {matched} (rtol={tolerance_rtol}, atol={tolerance_atol})")
    return {
        "matched": matched,
        "mean_diff": float(mean_diff),
        "max_diff": float(max_diff),
        "max_rel_diff": float(max_rel_diff),
        "plot_path": plot_path,
    }


def _safe_to_tensor(x):
    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            return None
        x = x[0]
    if not isinstance(x, torch.Tensor):
        return None
    return x


def _forward_model_in_batches(
    model: nn.Module,
    inputs: List[torch.Tensor],
    batch_size: int = 1,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if len(inputs) == 0:
        raise ValueError("inputs must not be empty")

    total = inputs[0].shape[0]
    outputs = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_inputs = [x[start:end] for x in inputs]
            outputs.append(model(*batch_inputs))
    return torch.cat(outputs, dim=0)


def analyze_quant_drop(
    float_model: nn.Module,
    quant_model: nn.Module,
    test_inputs: List[torch.Tensor],
    test_labels: torch.Tensor,
    dataset: str,
    enable_plot: bool = True,
    plot_prefix: str = "bert_quant_drop",
    eval_batch_size: int = 1,
):
    print("\n" + "=" * 80)
    print("Quantization Drop Analysis")
    print("=" * 80)

    module_specs = [
        ("embeddings_norm", "bert.embeddings.norm"),
        ("encoder_layer0", "bert.encoder.layers.0"),
        ("encoder_layer1", "bert.encoder.layers.1"),
        ("pooler_dense", "bert.pooler.dense"),
        ("classifier", "classifier"),
    ]

    def resolve_module(root: nn.Module, path: str) -> nn.Module:
        current = root
        for p in path.split("."):
            if p.isdigit():
                current = current[int(p)]
            else:
                current = getattr(current, p)
        return current

    float_acts: Dict[str, List[torch.Tensor]] = {}
    quant_acts: Dict[str, List[torch.Tensor]] = {}
    hooks = []

    for key, path in module_specs:
        float_mod = resolve_module(float_model, path)
        quant_mod = resolve_module(quant_model, path)

        def make_hook(container: Dict[str, List[torch.Tensor]], name: str):
            def _hook(_m, _inp, out):
                out_t = _safe_to_tensor(out)
                if out_t is not None:
                    container.setdefault(name, []).append(out_t.detach().cpu())
            return _hook

        hooks.append(float_mod.register_forward_hook(make_hook(float_acts, key)))
        hooks.append(quant_mod.register_forward_hook(make_hook(quant_acts, key)))

    float_logits = _forward_model_in_batches(float_model, test_inputs, batch_size=eval_batch_size)
    quant_logits = _forward_model_in_batches(quant_model, test_inputs, batch_size=eval_batch_size)

    for h in hooks:
        h.remove()

    float_acts_cat = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in float_acts.items()
        if len(chunks) > 0
    }
    quant_acts_cat = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in quant_acts.items()
        if len(chunks) > 0
    }

    pred_float = float_logits.argmax(-1)
    pred_quant = quant_logits.argmax(-1)
    disagreements = pred_float != pred_quant
    wrong_quant = pred_quant != test_labels
    wrong_float = pred_float != test_labels

    prob_float = torch.softmax(float_logits, dim=-1)
    prob_quant = torch.softmax(quant_logits, dim=-1)
    conf_float = prob_float.max(dim=-1).values
    conf_quant = prob_quant.max(dim=-1).values

    print(f"dataset={dataset}, batch={test_labels.shape[0]}")
    print(f"disagreement count (float vs quant): {int(disagreements.sum().item())}")
    print(f"float wrong count: {int(wrong_float.sum().item())}")
    print(f"quant wrong count: {int(wrong_quant.sum().item())}")
    print(f"mean confidence float: {conf_float.mean().item():.4f}")
    print(f"mean confidence quant: {conf_quant.mean().item():.4f}")

    layer_mae = {}
    layer_max = {}
    for key, _ in module_specs:
        if key not in float_acts_cat or key not in quant_acts_cat:
            continue
        f = float_acts_cat[key]
        q = quant_acts_cat[key]
        if f.shape != q.shape:
            continue
        d = torch.abs(f - q)
        layer_mae[key] = float(d.mean().item())
        layer_max[key] = float(d.max().item())
        print(f"layer={key:16s} mae={layer_mae[key]:.6f} max={layer_max[key]:.6f}")

    logits_cmp = compare_tensors_with_plot(
        float_logits,
        quant_logits,
        name="BERT logits float vs quant",
        enable_plot=enable_plot,
        plot_filename=f"{plot_prefix}_logits.png",
    )

    layer_plot_path = None
    if enable_plot and len(layer_mae) > 0:
        try:
            import matplotlib.pyplot as plt

            names = list(layer_mae.keys())
            mae_vals = [layer_mae[n] for n in names]
            max_vals = [layer_max[n] for n in names]

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].bar(names, mae_vals, color="#4e79a7")
            axes[0].set_title("Layer-wise MAE (float vs quant)")
            axes[0].set_ylabel("MAE")
            axes[0].grid(True, axis="y", alpha=0.3)

            axes[1].bar(names, max_vals, color="#e15759")
            axes[1].set_title("Layer-wise Max Abs Diff")
            axes[1].set_ylabel("Max |diff|")
            axes[1].grid(True, axis="y", alpha=0.3)

            for ax in axes:
                ax.tick_params(axis="x", rotation=20)

            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            layer_plot_path = f"{plot_prefix}_layerwise_{timestamp}.png"
            plt.savefig(layer_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✓ Layer-wise plot saved to: {layer_plot_path}")
        except ImportError:
            print("Warning: matplotlib not available, skip layer-wise plot")

    return {
        "disagreement_count": int(disagreements.sum().item()),
        "float_wrong_count": int(wrong_float.sum().item()),
        "quant_wrong_count": int(wrong_quant.sum().item()),
        "mean_conf_float": float(conf_float.mean().item()),
        "mean_conf_quant": float(conf_quant.mean().item()),
        "layer_mae": layer_mae,
        "layer_max": layer_max,
        "logits_cmp": logits_cmp,
        "layer_plot_path": layer_plot_path,
    }


def _resolve_module_by_path(root: nn.Module, path: str) -> nn.Module:
    current = root
    for p in path.split("."):
        if p.isdigit():
            current = current[int(p)]
        else:
            current = getattr(current, p)
    return current


def _set_module_by_path(root: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def _restore_float_modules(
    quant_model: nn.Module,
    float_model: nn.Module,
    module_paths: List[str],
) -> List[str]:
    restored = []
    for path in module_paths:
        try:
            float_submodule = _resolve_module_by_path(float_model, path)
            _set_module_by_path(quant_model, path, float_submodule)
            restored.append(path)
        except Exception as e:
            print(f"[Warning] failed to restore float module path={path}: {e}")
    return restored


def analyze_first_layer_internal_ops(
    float_layer: nn.Module,
    quant_layer: nn.Module,
    test_inputs: Tuple[torch.Tensor, torch.Tensor],
    enable_plot: bool = True,
    plot_prefix: str = "bert_first_layer_ops",
    output_dir: Optional[str] = None,
):
    print("\n" + "=" * 80)
    print("First-Layer Internal Operator Analysis")
    print("=" * 80)

    module_specs = [
        ("linear_q", "attention.linear_q"),
        ("linear_k", "attention.linear_k"),
        ("linear_v", "attention.linear_v"),
        ("matmul1", "attention.matmul1"),
        ("softmax", "attention.softmax"),
        ("matmul2", "attention.matmul2"),
        ("linear_out", "attention.linear_out"),
        ("add1", "add1"),
        ("norm1", "norm1"),
        ("ffn_fc1", "ffn.fc1"),
        ("ffn_gelu", "ffn.activation"),
        ("ffn_fc2", "ffn.fc2"),
        ("add2", "add2"),
        ("norm2", "norm2"),
    ]

    float_acts: Dict[str, torch.Tensor] = {}
    quant_acts: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(container: Dict[str, torch.Tensor], name: str):
        def _hook(_m, _inp, out):
            out_t = _safe_to_tensor(out)
            if out_t is not None:
                container[name] = out_t.detach().cpu()
        return _hook

    for key, path in module_specs:
        try:
            float_mod = _resolve_module_by_path(float_layer, path)
            quant_mod = _resolve_module_by_path(quant_layer, path)
        except Exception as e:
            print(f"[Warning] skip op={key}, path={path}, reason={e}")
            continue

        hooks.append(float_mod.register_forward_hook(make_hook(float_acts, key)))
        hooks.append(quant_mod.register_forward_hook(make_hook(quant_acts, key)))

    with torch.no_grad():
        float_out = float_layer(*test_inputs)
        quant_out = quant_layer(*test_inputs)

    for h in hooks:
        h.remove()

    op_stats = {}
    for key, _ in module_specs:
        if key not in float_acts or key not in quant_acts:
            continue

        f = float_acts[key]
        q = quant_acts[key]
        if f.shape != q.shape:
            print(f"[Warning] op={key} shape mismatch: {f.shape} vs {q.shape}")
            continue

        d = torch.abs(f - q)
        mae = float(d.mean().item())
        max_diff = float(d.max().item())
        op_stats[key] = {
            "mae": mae,
            "max": max_diff,
            "shape": list(f.shape),
        }
        print(f"op={key:12s} mae={mae:.6f} max={max_diff:.6f} shape={tuple(f.shape)}")

        compare_tensors_with_plot(
            f,
            q,
            name=f"First-Layer Op {key} (float vs quant)",
            enable_plot=enable_plot,
            plot_filename=(
                os.path.join(output_dir, f"{plot_prefix}_{key}.png")
                if output_dir is not None
                else f"{plot_prefix}_{key}.png"
            ),
        )

    out_diff = torch.abs(float_out - quant_out)
    output_stats = {
        "mae": float(out_diff.mean().item()),
        "max": float(out_diff.max().item()),
    }
    print(
        f"first-layer-output mae={output_stats['mae']:.6f} "
        f"max={output_stats['max']:.6f}"
    )

    summary_plot_path = None
    if enable_plot and len(op_stats) > 0:
        try:
            import matplotlib.pyplot as plt

            names = list(op_stats.keys())
            mae_vals = [op_stats[n]["mae"] for n in names]
            max_vals = [op_stats[n]["max"] for n in names]

            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes[0].bar(names, mae_vals, color="#4e79a7")
            axes[0].set_title("First-Layer Operator MAE (float vs quant)")
            axes[0].set_ylabel("MAE")
            axes[0].grid(True, axis="y", alpha=0.3)

            axes[1].bar(names, max_vals, color="#e15759")
            axes[1].set_title("First-Layer Operator Max Abs Diff")
            axes[1].set_ylabel("Max |diff|")
            axes[1].grid(True, axis="y", alpha=0.3)

            for ax in axes:
                ax.tick_params(axis="x", rotation=25)

            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_dir is not None:
                summary_plot_path = os.path.join(output_dir, f"{plot_prefix}_summary_{timestamp}.png")
            else:
                summary_plot_path = f"{plot_prefix}_summary_{timestamp}.png"
            plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✓ First-layer operator summary plot saved to: {summary_plot_path}")
        except ImportError:
            print("Warning: matplotlib not available, skip first-layer operator summary plot")

    return {
        "output_stats": output_stats,
        "op_stats": op_stats,
        "summary_plot_path": summary_plot_path,
    }


def _get_quant_config(run_compile: bool):
    quant_config = get_vit_optimized_config()
    if run_compile:
        set_extra_compile_param_for_config(quant_config)
    # if use lut, enable this
    set_extra_compile_param_for_config(quant_config)
    return quant_config


def _build_custom_bert_from_hf(hf_model: nn.Module) -> BertForSequenceClassification:
    cfg = hf_model.config
    model = BertForSequenceClassification(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        num_layers=cfg.num_hidden_layers,
        max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size,
        num_labels=cfg.num_labels,
    ).eval()
    return model


def _load_custom_bert_weights_from_hf(custom_model: BertForSequenceClassification, hf_model: nn.Module):
    custom_model.bert.embeddings.word_embeddings.weight.data = hf_model.bert.embeddings.word_embeddings.weight.data
    custom_model.bert.embeddings.position_embeddings.weight.data = hf_model.bert.embeddings.position_embeddings.weight.data
    custom_model.bert.embeddings.token_type_embeddings.weight.data = hf_model.bert.embeddings.token_type_embeddings.weight.data
    custom_model.bert.embeddings.norm.weight.data = hf_model.bert.embeddings.LayerNorm.weight.data
    custom_model.bert.embeddings.norm.bias.data = hf_model.bert.embeddings.LayerNorm.bias.data
    custom_model.bert.embeddings.norm.eps = hf_model.bert.embeddings.LayerNorm.eps

    for i in range(len(custom_model.bert.encoder.layers)):
        dst = custom_model.bert.encoder.layers[i]
        src = hf_model.bert.encoder.layer[i]

        dst.attention.linear_q.weight.data = src.attention.self.query.weight.data
        dst.attention.linear_q.bias.data = src.attention.self.query.bias.data
        dst.attention.linear_k.weight.data = src.attention.self.key.weight.data
        dst.attention.linear_k.bias.data = src.attention.self.key.bias.data
        dst.attention.linear_v.weight.data = src.attention.self.value.weight.data
        dst.attention.linear_v.bias.data = src.attention.self.value.bias.data
        dst.attention.linear_out.weight.data = src.attention.output.dense.weight.data
        dst.attention.linear_out.bias.data = src.attention.output.dense.bias.data

        dst.norm1.weight.data = src.attention.output.LayerNorm.weight.data
        dst.norm1.bias.data = src.attention.output.LayerNorm.bias.data
        dst.norm1.eps = src.attention.output.LayerNorm.eps

        dst.ffn.fc1.weight.data = src.intermediate.dense.weight.data
        dst.ffn.fc1.bias.data = src.intermediate.dense.bias.data
        dst.ffn.fc2.weight.data = src.output.dense.weight.data
        dst.ffn.fc2.bias.data = src.output.dense.bias.data

        dst.norm2.weight.data = src.output.LayerNorm.weight.data
        dst.norm2.bias.data = src.output.LayerNorm.bias.data
        dst.norm2.eps = src.output.LayerNorm.eps

    custom_model.bert.pooler.dense.weight.data = hf_model.bert.pooler.dense.weight.data
    custom_model.bert.pooler.dense.bias.data = hf_model.bert.pooler.dense.bias.data

    custom_model.classifier.weight.data = hf_model.classifier.weight.data
    custom_model.classifier.bias.data = hf_model.classifier.bias.data


def test_first_bert_layer(
    model_name: str = "bert-tiny-cls",
    sample_batch_size: int = 16,
    test_batch_size: int = 64,
    run_compile: bool = False,
    analyze_first_layer: bool = False,
    first_layer_plot: bool = False,
    first_layer_prefix: str = "bert_first_layer_drop",
    analyze_first_layer_ops: bool = False,
    mute_module: Optional[str] = None,
):
    """Test quantization for first BERT encoder layer (layer-0)."""
    if model_name not in BERT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BERT_CONFIGS.keys())}")

    model_path = BERT_CONFIGS[model_name]["model_path"]
    max_length = BERT_CONFIGS[model_name]["max_length"]

    print("\n" + "=" * 60)
    print(f"Testing First BERT Layer ({model_name})")
    print("=" * 60)
    print(f"[Config] model_path={model_path}")
    print(f"[Config] max_length={max_length}")

    hf_model, tokenizer = _load_hf_model_and_tokenizer(model_path)
    base_model = _build_custom_bert_from_hf(hf_model)
    _load_custom_bert_weights_from_hf(base_model, hf_model)

    calib_texts = _build_calibration_text_samples(sample_batch_size, max_length)
    test_texts = _build_text_samples(test_batch_size)

    calib_batch = _tokenize_batch(tokenizer, calib_texts, max_length=max_length)
    test_batch = _tokenize_batch(tokenizer, test_texts, max_length=max_length)

    calib_inputs = _prepare_first_layer_inputs(base_model, calib_batch)
    test_inputs = _prepare_first_layer_inputs(base_model, test_batch)

    module = BertLayer(
        hidden_size=hf_model.config.hidden_size,
        num_heads=hf_model.config.num_attention_heads,
        intermediate_size=hf_model.config.intermediate_size,
    ).eval()
    module.load_state_dict(base_model.bert.encoder.layers[0].state_dict())

    quant_config = _get_quant_config(run_compile)
    qmodule = replace_module_with_quantized(module, config=quant_config)

    if mute_module is not None and len(mute_module.strip()) > 0:
        mute_module = mute_module.strip()
        if mute_module not in FIRST_LAYER_MUTABLE_MODULES:
            raise ValueError(
                f"Unsupported --mute-module={mute_module}. "
                f"Available: {FIRST_LAYER_MUTABLE_MODULES}"
            )
        float_submodule = _resolve_module_by_path(module, mute_module)
        _set_module_by_path(qmodule, mute_module, float_submodule)
        print(f"[Ablation] mute quantization for first-layer module: {mute_module}")

    calibrator = Calibrator(qmodule, list(calib_inputs))
    calibrator.calibrate()

    if run_compile:
        batch = 1
        print("\nCompiling first BERT layer with allo...")
        llvm_mod = allo.frontend.from_pytorch_vivado(
            qmodule,
            example_inputs=[test_inputs[0][:batch], test_inputs[1][:batch]],
            leaf_modules=[QLinear, IntLayerNorm, IntSoftmax, IntSoftmaxWithMask, IntGELU, QAdd, QMatMul, QMatMulIsqrtD],
            quant_config=quant_config,
            verbose=False,
            project="pynq_vivado_bert_first_layer.prj",
            mode="default",
        )
        print("    Compilation completed!")
        return llvm_mod

    calibrator.enable_fakequant()
    calibrator.enable_lut_inference()

    run_output_dir = None
    if first_layer_plot and (analyze_first_layer or analyze_first_layer_ops):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mute_module is not None and len(mute_module.strip()) > 0:
            run_tag = f"mute_{mute_module.replace('.', '_')}"
        else:
            run_tag = "baseline"
        run_output_dir = f"{first_layer_prefix}_{run_tag}_{timestamp}"
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"[Output] first-layer plots will be saved to: {run_output_dir}")

    with torch.no_grad():
        golden = module(*test_inputs)
        res_fake = qmodule(*test_inputs)

    _print_diff_stats("First BERT Layer", golden, res_fake)

    analysis_result = None
    if analyze_first_layer:
        analysis_result = compare_tensors_with_plot(
            golden,
            res_fake,
            name="BERT First Layer Output (float vs quant)",
            enable_plot=first_layer_plot,
            plot_filename=(
                os.path.join(run_output_dir, f"{first_layer_prefix}.png")
                if run_output_dir is not None
                else f"{first_layer_prefix}.png"
            ),
        )

        token_mae = torch.mean(torch.abs(golden - res_fake), dim=-1)
        print(f"    Token MAE mean: {token_mae.mean().item():.6f}")
        print(f"    Token MAE max:  {token_mae.max().item():.6f}")

        if first_layer_plot:
            try:
                import matplotlib.pyplot as plt

                token_mae_np = token_mae.detach().cpu().numpy()
                fig, ax = plt.subplots(figsize=(12, 4))
                im = ax.imshow(token_mae_np, aspect="auto", interpolation="nearest", cmap="viridis")
                ax.set_title("First Layer Token-wise MAE (batch x seq)")
                ax.set_xlabel("Sequence Index")
                ax.set_ylabel("Batch Index")
                plt.colorbar(im, ax=ax, label="MAE")
                plt.tight_layout()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if run_output_dir is not None:
                    heatmap_path = os.path.join(run_output_dir, f"{first_layer_prefix}_token_mae_{timestamp}.png")
                else:
                    heatmap_path = f"{first_layer_prefix}_token_mae_{timestamp}.png"
                plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"✓ First-layer token MAE heatmap saved to: {heatmap_path}")
            except ImportError:
                print("Warning: matplotlib not available, skip first-layer heatmap")

    op_analysis_result = None
    if analyze_first_layer_ops:
        op_analysis_result = analyze_first_layer_internal_ops(
            float_layer=module,
            quant_layer=qmodule,
            test_inputs=test_inputs,
            enable_plot=first_layer_plot,
            plot_prefix=f"{first_layer_prefix}_ops",
            output_dir=run_output_dir,
        )

    return {
        "analysis": analysis_result,
        "op_analysis": op_analysis_result,
    }


def test_calibrate_bert_model(
    model_name: str = "bert-tiny-cls",
    sample_batch_size: int = 16,
    test_batch_size: int = 64,
    run_compile: bool = False,
    dataset: str = "mnli",
    mnli_root: str = DEFAULT_MNLI_ROOT,
    mnli_split: str = "dev_matched",
    qnli_root: str = DEFAULT_QNLI_ROOT,
    qnli_split: str = "dev",
    eval_size: int = 200,
    analyze_drop_flag: bool = False,
    analysis_plot: bool = True,
    analysis_prefix: str = "bert_quant_drop",
    float_module_paths: Optional[List[str]] = None,
    eval_batch_size: int = 1,
):
    """Test quantization for full BERT sequence-classification model."""
    if model_name not in BERT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BERT_CONFIGS.keys())}")

    model_path = BERT_CONFIGS[model_name]["model_path"]
    max_length = BERT_CONFIGS[model_name]["max_length"]

    print("\n" + "=" * 60)
    print(f"Testing Full BERT Model Quantization ({model_name})")
    print("=" * 60)
    print(f"[Config] model_path={model_path}")
    print(f"[Config] max_length={max_length}")

    hf_model, tokenizer = _load_hf_model_and_tokenizer(model_path)
    module = _build_custom_bert_from_hf(hf_model)
    _load_custom_bert_weights_from_hf(module, hf_model)

    label_to_id = _build_label_to_id(hf_model, task=dataset)
    if dataset == "mnli":
        premises, hypotheses, true_labels = _load_mnli_split(
            mnli_root=mnli_root,
            split=mnli_split,
            limit=eval_size,
            label_to_id=label_to_id,
        )
        dataset_root_print = mnli_root
        dataset_split_print = mnli_split
        calib_premises, calib_hypotheses, _ = _load_mnli_split(
            mnli_root=mnli_root,
            split="train",
            limit=sample_batch_size,
            label_to_id=label_to_id,
        )
    elif dataset == "qnli":
        premises, hypotheses, true_labels = _load_qnli_split(
            qnli_root=qnli_root,
            split=qnli_split,
            limit=eval_size,
            label_to_id=label_to_id,
        )
        dataset_root_print = qnli_root
        dataset_split_print = qnli_split
        calib_premises, calib_hypotheses, _ = _load_qnli_split(
            qnli_root=qnli_root,
            split="train",
            limit=sample_batch_size,
            label_to_id=label_to_id,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Available: ['mnli', 'qnli']")

    effective_eval_size = eval_size if eval_size > 0 else test_batch_size
    if effective_eval_size <= 0:
        effective_eval_size = len(true_labels)
    eval_count = min(effective_eval_size, len(true_labels))

    test_premises = premises[:eval_count]
    test_hypotheses = hypotheses[:eval_count]
    test_labels = torch.tensor(true_labels[:eval_count], dtype=torch.long)

    print(f"[{dataset.upper()}] root={dataset_root_print}")
    print(f"[{dataset.upper()}] split={dataset_split_print}")
    print(f"[{dataset.upper()}] loaded_samples={len(true_labels)}, eval_samples={eval_count}")
    print(f"[{dataset.upper()}] calibration_split=train, calibration_samples={len(calib_premises)}")
    print(f"[Eval] batch_size={eval_batch_size}")

    calib_batch = _tokenize_batch(
        tokenizer,
        calib_premises,
        max_length=max_length,
        pair_texts=calib_hypotheses,
    )
    test_batch = _tokenize_batch(
        tokenizer,
        test_premises,
        max_length=max_length,
        pair_texts=test_hypotheses,
    )

    calib_inputs = [
        calib_batch["input_ids"],
        calib_batch["attention_mask"],
        calib_batch["token_type_ids"],
    ]
    test_inputs = [
        test_batch["input_ids"],
        test_batch["attention_mask"],
        test_batch["token_type_ids"],
    ]

    quant_config = _get_quant_config(run_compile)
    qmodule = replace_module_with_quantized(module, config=quant_config)

    restored_paths = []
    if float_module_paths is not None and len(float_module_paths) > 0:
        restored_paths = _restore_float_modules(qmodule, module, float_module_paths)
        if len(restored_paths) > 0:
            print(f"[Ablation] keep float modules: {restored_paths}")

    calibrator = Calibrator(qmodule, calib_inputs)
    calibrator.calibrate()

    if run_compile:
        batch = 1
        print("\nCompiling full BERT model with allo (experimental)...")
        llvm_mod = allo.frontend.from_pytorch_vivado(
            qmodule,
            example_inputs=[
                test_inputs[0][:batch],
                test_inputs[1][:batch],
                test_inputs[2][:batch],
            ],
            leaf_modules=[BertGetFirstToken, QLinear, IntLayerNorm, IntSoftmax, IntGELU, QAdd, QMatMul, QMatMulIsqrtD],
            quant_config=quant_config,
            verbose=False,
            project="pynq_vivado_bert_model.prj",
            mode="default",
        )
        print("    Compilation completed!")
        return llvm_mod

    calibrator.enable_fakequant()
    calibrator.enable_lut_inference()

    golden = _forward_model_in_batches(module, test_inputs, batch_size=eval_batch_size)
    res_fake = _forward_model_in_batches(qmodule, test_inputs, batch_size=eval_batch_size)

    _print_diff_stats("Full BERT Model", golden, res_fake)

    pred_float = golden.argmax(-1)
    pred_quant = res_fake.argmax(-1)

    total = golden.shape[0]
    float_top1_match = torch.sum(pred_float == test_labels).item()
    quant_top1_match = torch.sum(pred_quant == test_labels).item()
    float_quant_match = torch.sum(pred_float == pred_quant).item()

    float_top1_acc = 100.0 * float_top1_match / total
    quant_top1_acc = 100.0 * quant_top1_match / total
    acc_drop = float_top1_acc - quant_top1_acc

    print(f"    Top1 Acc (float vs label): {float_top1_match}/{total} ({float_top1_acc:.1f}%)")
    print(f"    Top1 Acc (quant vs label): {quant_top1_match}/{total} ({quant_top1_acc:.1f}%)")
    print(f"    Accuracy Drop (float-quant): {acc_drop:.2f}%")
    print(f"    Top1 Match (quant vs float): {float_quant_match}/{total} ({100.0 * float_quant_match / total:.1f}%)")

    analysis_result = None
    if analyze_drop_flag:
        analysis_result = analyze_quant_drop(
            float_model=module,
            quant_model=qmodule,
            test_inputs=test_inputs,
            test_labels=test_labels,
            dataset=dataset,
            enable_plot=analysis_plot,
            plot_prefix=analysis_prefix,
            eval_batch_size=eval_batch_size,
        )

    return {
        "top1_match": float_quant_match,
        "total": total,
        "float_top1_match": float_top1_match,
        "quant_top1_match": quant_top1_match,
        "float_top1_acc": float_top1_acc,
        "quant_top1_acc": quant_top1_acc,
        "acc_drop": acc_drop,
        "analysis": analysis_result,
        "restored_float_modules": restored_paths,
    }


def run_quant_ablation_suite(
    model_name: str,
    sample_batch_size: int,
    test_batch_size: int,
    run_compile: bool,
    dataset: str,
    mnli_root: str,
    mnli_split: str,
    qnli_root: str,
    qnli_split: str,
    eval_size: int,
):
    variants = [
        ("full_quant", []),
        ("classifier_float", ["classifier"]),
        ("pooler_classifier_float", ["bert.pooler.dense", "classifier"]),
        ("trunk_float_classifier_quant", ["bert"]),
    ]

    summary = []
    print("\n" + "=" * 60)
    print("Running Quantization Ablation Suite")
    print("=" * 60)
    for name, keep_float in variants:
        print(f"\n[Ablation] variant={name}, keep_float={keep_float}")
        res = test_calibrate_bert_model(
            model_name=model_name,
            sample_batch_size=sample_batch_size,
            test_batch_size=test_batch_size,
            run_compile=run_compile,
            dataset=dataset,
            mnli_root=mnli_root,
            mnli_split=mnli_split,
            qnli_root=qnli_root,
            qnli_split=qnli_split,
            eval_size=eval_size,
            analyze_drop_flag=False,
            analysis_plot=False,
            analysis_prefix=f"ablation_{name}",
            float_module_paths=keep_float,
        )
        summary.append(
            {
                "variant": name,
                "float_acc": res["float_top1_acc"],
                "quant_acc": res["quant_top1_acc"],
                "acc_drop": res["acc_drop"],
                "restored": res.get("restored_float_modules", []),
            }
        )

    print("\n" + "-" * 60)
    print("Ablation Summary")
    print("-" * 60)
    for row in summary:
        print(
            f"{row['variant']:28s} "
            f"float={row['float_acc']:.2f}% "
            f"quant={row['quant_acc']:.2f}% "
            f"drop={row['acc_drop']:.2f}% "
            f"keep_float={row['restored']}"
        )

    return summary


def main():
    parser = argparse.ArgumentParser(description="BERT INT8 Quantization Test")
    parser.add_argument(
        "--model",
        type=str,
        default="bert-tiny-cls",
        choices=list(BERT_CONFIGS.keys()),
        help="BERT model variant to test",
    )
    parser.add_argument(
        "--sample-batch",
        type=int,
        default=16,
        help="Batch size for calibration samples",
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        default=64,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        choices=["mnli", "qnli"],
        help="Dataset/task for full-model evaluation",
    )
    parser.add_argument(
        "--mnli-root",
        type=str,
        default=DEFAULT_MNLI_ROOT,
        help="Local MultiNLI root directory",
    )
    parser.add_argument(
        "--mnli-split",
        type=str,
        default="dev_matched",
        choices=list(MNLI_SPLIT_TO_FILE.keys()),
        help="MultiNLI split for full-model evaluation",
    )
    parser.add_argument(
        "--qnli-root",
        type=str,
        default=DEFAULT_QNLI_ROOT,
        help="Local QNLI root directory",
    )
    parser.add_argument(
        "--qnli-split",
        type=str,
        default="dev",
        choices=list(QNLI_SPLIT_TO_FILE.keys()),
        help="QNLI split for full-model evaluation",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=200,
        help="Number of MNLI samples to evaluate on full-model path",
    )
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=1,
        help="Batch size for full-model evaluation. Default 1 avoids batch-varying attention-mask issues.",
    )
    parser.add_argument(
        "--analyze-drop",
        action="store_true",
        help="Run layer-wise float-vs-quant drop analysis after evaluation",
    )
    parser.add_argument(
        "--analysis-plot",
        action="store_true",
        help="Enable plot generation in drop analysis (requires matplotlib)",
    )
    parser.add_argument(
        "--analysis-prefix",
        type=str,
        default="bert_quant_drop",
        help="Prefix of generated analysis plot files",
    )
    parser.add_argument(
        "--float-modules",
        type=str,
        nargs="*",
        default=[],
        help="Module paths to keep in float (e.g. classifier bert.pooler.dense bert)",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run built-in partial-quantization ablation variants",
    )
    parser.add_argument(
        "--analyze-first-layer",
        action="store_true",
        help="Run detailed first-layer float-vs-quant analysis",
    )
    parser.add_argument(
        "--first-layer-plot",
        action="store_true",
        help="Enable first-layer plot generation (requires matplotlib)",
    )
    parser.add_argument(
        "--first-layer-prefix",
        type=str,
        default="bert_first_layer_drop",
        help="Prefix of generated first-layer analysis plots",
    )
    parser.add_argument(
        "--analyze-first-layer-ops",
        action="store_true",
        help="Analyze internal operators inside first BERT layer and generate per-op plots",
    )
    parser.add_argument(
        "--mute-module",
        type=str,
        default="",
        help=(
            "In first-layer path, keep one selected submodule in float while quantizing others. "
            f"Available: {FIRST_LAYER_MUTABLE_MODULES}"
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with allo instead of running inference",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="model",
        choices=["first-layer", "model"],
        help="Select test target: first-layer or full model",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("    BERT INT8 Quantization Test")
    print("=" * 60)

    if args.test == "first-layer":
        test_first_bert_layer(
            model_name=args.model,
            sample_batch_size=args.sample_batch,
            test_batch_size=args.test_batch,
            run_compile=args.compile,
            analyze_first_layer=args.analyze_first_layer,
            first_layer_plot=args.first_layer_plot,
            first_layer_prefix=args.first_layer_prefix,
            analyze_first_layer_ops=args.analyze_first_layer_ops,
            mute_module=args.mute_module,
        )
    else:
        expected_model_by_dataset = {
            "mnli": "bert-tiny-cls",
            "qnli": "bert-tiny-qnli",
        }
        expected_model = expected_model_by_dataset[args.dataset]
        if args.model != expected_model:
            print(
                f"[Warning] dataset={args.dataset} is usually paired with model={expected_model}, "
                f"but got model={args.model}."
            )

        if args.run_ablation:
            run_quant_ablation_suite(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
                dataset=args.dataset,
                mnli_root=args.mnli_root,
                mnli_split=args.mnli_split,
                qnli_root=args.qnli_root,
                qnli_split=args.qnli_split,
                eval_size=args.eval_size,
            )
        else:
            test_calibrate_bert_model(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
                dataset=args.dataset,
                mnli_root=args.mnli_root,
                mnli_split=args.mnli_split,
                qnli_root=args.qnli_root,
                qnli_split=args.qnli_split,
                eval_size=args.eval_size,
                analyze_drop_flag=args.analyze_drop,
                analysis_plot=args.analysis_plot,
                analysis_prefix=args.analysis_prefix,
                float_module_paths=args.float_modules,
                eval_batch_size=args.eval_batch,
            )


if __name__ == "__main__":
    main()
