"""
ViT INT8 Quantization Test (Refactored)

This module tests INT8 quantization for Vision Transformer (ViT) models.
Uses allo.ops.vit for model structure and loads real HuggingFace models.

For deprecated manual calibration methods, see vit_int8_legacy.py
"""

import allo
import torch
import torch.nn as nn
import numpy as np
import os
import time
from datetime import datetime

# Use allo.ops.vit structures to ensure compatibility
from allo.ops.vit import (
    ViTImgCls, ViTBlock, ViTGetFirstToken, ViTTokenExpand, MatMulIsqrtD, Add,
)

# Import quantization utilities from quant_config (no duplicate definitions!)
from allo.quant.quant_config import (
    replace_module_with_quantized,
    Calibrator,
    get_default_config,
    get_vit_optimized_config,
    set_extra_compile_param_for_config,
)
from allo.quant.quant_modules import (
    QLinear, QConv2d, IntGELU, IntSoftmax, IntLayerNorm, QAdd, QMatMul, QMatMulIsqrtD,
)

from utils import replace_vit_with_hf_vit, get_imagenet_test_data


# ==============================================================================
# ViT Model Configurations
# ==============================================================================

VIT_CONFIGS = {
    "deit-tiny": {
        "n_embd": 192,
        "n_head": 3,
        "n_layers": 12,
        "model_path": "/root/data/models/deit-tiny-patch16-224",
    },
    "deit-small": {
        "n_embd": 384,
        "n_head": 6,
        "n_layers": 12,
        "model_path": "/root/data/models/deit-small-patch16-224",
    },
    "deit-base": {
        "n_embd": 768,
        "n_head": 12,
        "n_layers": 12,
        "model_path": "/root/data/models/deit-base-patch16-224",
    },
    "vit-base": {
        "n_embd": 768,
        "n_head": 12,
        "n_layers": 12,
        "model_path": "/root/data/models/vit-base-patch16-224",
    },
}


class TwoInputsModule(nn.Module):
    def __init__(self, matmul):
        super().__init__()
        self.matmul = matmul

    def forward(self, q, k_t):
        return self.matmul(q, k_t)


class SingleInputModule(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x)


class FirstLayerNormLinearModule(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.linear_q = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.linear_q(self.norm(x))


class FirstLayerNormQKMatMulModule(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.linear_q = nn.Linear(n_embd, n_embd)
        self.linear_k = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.matmul = MatMulIsqrtD(self.head_dim)

    def forward(self, x):
        x = self.norm(x)
        q = self.linear_q(x)
        k = self.linear_k(x)

        bsz, seq_len, _ = q.shape
        q = q.reshape(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k_t = k.transpose(-1, -2)
        return self.matmul(q, k_t)

def load_vit_block_from_hf(blk: ViTBlock, hf_vit, layer_idx: int = 0) -> ViTBlock:
    """Load the weights of a HuggingFace ViT encoder block into a ViTBlock."""
    layer = hf_vit.vit.encoder.layer[layer_idx]
    blk.attention.linear_q.weight.data = layer.attention.attention.query.weight.data
    blk.attention.linear_q.bias.data = layer.attention.attention.query.bias.data
    blk.attention.linear_k.weight.data = layer.attention.attention.key.weight.data
    blk.attention.linear_k.bias.data = layer.attention.attention.key.bias.data
    blk.attention.linear_v.weight.data = layer.attention.attention.value.weight.data
    blk.attention.linear_v.bias.data = layer.attention.attention.value.bias.data
    blk.attention.linear_out.weight.data = layer.attention.output.dense.weight.data
    blk.attention.linear_out.bias.data = layer.attention.output.dense.bias.data
    blk.ffn.fc1.weight.data = layer.intermediate.dense.weight.data
    blk.ffn.fc1.bias.data = layer.intermediate.dense.bias.data
    blk.ffn.fc2.weight.data = layer.output.dense.weight.data
    blk.ffn.fc2.bias.data = layer.output.dense.bias.data
    blk.norm1.weight.data = layer.layernorm_before.weight.data
    blk.norm1.bias.data = layer.layernorm_before.bias.data
    blk.norm1.eps = layer.layernorm_before.eps
    blk.norm2.weight.data = layer.layernorm_after.weight.data
    blk.norm2.bias.data = layer.layernorm_after.bias.data
    blk.norm2.eps = layer.layernorm_after.eps
    return blk


def compare_tensors_with_plot(
    golden, 
    ref, 
    name: str = "Tensor Comparison",
    tolerance_rtol: float = 1e-5, 
    tolerance_atol: float = 1e-6,
    enable_plot: bool = True,
    plot_filename: str = "tensor_comparison.png"
):
    """
    Compare two tensors and optionally generate visualization plots.
    
    Mimics the behavior of utils.compare_binary_files but works with torch tensors directly.
    Creates 4 subplots: data overlay, absolute difference, first 100 points detail, and scatter plot.
    
    Args:
        golden: Reference/golden output tensor
        ref: Test/quantized output tensor
        name: Name of the comparison (for display)
        tolerance_rtol: Relative tolerance for matching
        tolerance_atol: Absolute tolerance for matching
        enable_plot: Whether to generate visualization plot
        plot_filename: Name of the output plot file
    
    Returns:
        dict: Contains keys 'matched', 'mean_diff', 'max_diff', 'max_rel_diff'
    """
    # Convert to numpy for computation
    if isinstance(golden, torch.Tensor):
        golden = golden.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    
    # Flatten for analysis
    golden_flat = golden.flatten()
    ref_flat = ref.flatten()
    
    if len(golden_flat) != len(ref_flat):
        print(f"Error: Size mismatch - {golden_flat.shape} vs {ref_flat.shape}")
        return {'matched': False, 'error': 'Size mismatch'}
    
    # Compute statistics
    diff = np.abs(golden_flat - ref_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    max_rel_diff = np.max(diff / (np.abs(ref_flat) + 1e-8))
    
    # Print statistics
    print("\n" + "=" * 80)
    print(f"{name} - Tensor Comparison")
    print("=" * 80)
    print(f"Golden shape:  {golden.shape}")
    print(f"Ref shape:     {ref.shape}")
    print(f"Golden stats:  mean={golden_flat.mean():.6f}, std={golden_flat.std():.6f}")
    print(f"               min={golden_flat.min():.6f}, max={golden_flat.max():.6f}")
    print(f"Ref stats:     mean={ref_flat.mean():.6f}, std={ref_flat.std():.6f}")
    print(f"               min={ref_flat.min():.6f}, max={ref_flat.max():.6f}")
    
    print(f"\n{'='*80}")
    print(f"差异统计 (Error Statistics)")
    print(f"{'='*80}")
    print(f"  绝对差异 - mean: {mean_diff:.6f}, max: {max_diff:.6f}")
    print(f"  相对差异 - max: {max_rel_diff:.6f}")
    
    # Generate visualization if requested
    if enable_plot:
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Full data comparison (sampled)
            indices = np.arange(len(golden_flat))
            sample_step = max(1, len(golden_flat) // 1000)  # At most 1000 points
            sample_indices = indices[::sample_step]
            
            axes[0, 0].plot(sample_indices, golden_flat[::sample_step], label='Golden', alpha=0.7, linewidth=0.5)
            axes[0, 0].plot(sample_indices, ref_flat[::sample_step], label='Ref (Quantized)', alpha=0.7, linewidth=0.5)
            axes[0, 0].set_xlabel('Index')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].set_title('Full Output Comparison (Sampled)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Absolute difference curve
            axes[0, 1].plot(sample_indices, diff[::sample_step], color='red', alpha=0.7, linewidth=0.5)
            axes[0, 1].set_xlabel('Index')
            axes[0, 1].set_ylabel('Absolute Difference')
            axes[0, 1].set_title('Absolute Difference (|Golden - Ref|)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. First 100 points detail
            num_points = min(100, len(golden_flat))
            axes[1, 0].plot(np.arange(num_points), golden_flat[:num_points], label='Golden', marker='o', markersize=3, alpha=0.7)
            axes[1, 0].plot(np.arange(num_points), ref_flat[:num_points], label='Ref', marker='s', markersize=3, alpha=0.7)
            axes[1, 0].set_xlabel('Index')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('First 100 Points Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Scatter plot: Golden vs Ref
            axes[1, 1].scatter(golden_flat[::sample_step], ref_flat[::sample_step], alpha=0.5, s=1)
            axes[1, 1].plot([golden_flat.min(), golden_flat.max()], [golden_flat.min(), golden_flat.max()],
                           'r--', label='y = x', linewidth=1)
            axes[1, 1].set_xlabel('Golden')
            axes[1, 1].set_ylabel('Ref (Quantized)')
            axes[1, 1].set_title('Scatter Plot: Golden vs Ref')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"{plot_filename[:-4]}_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Comparison plot saved to: {plot_path}")
            plt.close()
            
        except ImportError:
            print("\n⚠ Warning: matplotlib not available, skipping visualization")
    
    # Check if tensors match within tolerance
    matched = np.allclose(golden_flat, ref_flat, rtol=tolerance_rtol, atol=tolerance_atol)
    if matched:
        print(f"\n✓ 匹配成功 (Match successful) - rtol={tolerance_rtol}, atol={tolerance_atol}")
    else:
        print(f"\n✗ 匹配失败 (Match failed) - rtol={tolerance_rtol}, atol={tolerance_atol}")
    
    return {
        'matched': matched,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'max_rel_diff': max_rel_diff,
        'plot_path': plot_path if enable_plot else None
    }



def _run_single_op_quant_test(
    op_name: str,
    module: nn.Module,
    calib_inputs,
    test_inputs,
    run_compile: bool,
    leaf_modules,
    project: str,
):
    """Shared flow for single-op quant test: calibrate, fakequant compare, optional compile compare."""
    quant_config = get_vit_optimized_config()
    if run_compile:
        set_extra_compile_param_for_config(quant_config)
    # for frontend validate lut
    set_extra_compile_param_for_config(quant_config)

    qmodule = replace_module_with_quantized(module, config=quant_config)
    calibrator = Calibrator(qmodule, calib_inputs)
    calibrator.calibrate()

    if not run_compile:
        calibrator.enable_fakequant()
        # for frontend validate lut. below is the same
        calibrator.enable_lut_inference()
        with torch.no_grad():
            if isinstance(test_inputs, (list, tuple)):
                golden = module(*test_inputs)
                res_fake = qmodule(*test_inputs)
            else:
                golden = module(test_inputs)
                res_fake = qmodule(test_inputs)

        mean_diff = torch.mean(torch.abs(golden - res_fake)).item()
        max_diff = torch.max(torch.abs(golden - res_fake)).item()

        print("\n" + "-" * 40)
        print(f"{op_name} Results (float vs fakequant):")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Max diff: {max_diff:.6f}")
        return None

    else:
        batch = 1
        if isinstance(calib_inputs, (list, tuple)):
            compile_inputs = [x[:batch] for x in calib_inputs]
        else:
            compile_inputs = [calib_inputs[:batch]]

        print(f"\nCompiling {op_name} with allo...")
        compile_start = time.perf_counter()
        llvm_mod = allo.frontend.from_pytorch_vivado(
            qmodule,
            example_inputs=compile_inputs,
            leaf_modules=leaf_modules,
            quant_config=quant_config,
            verbose=False,
            project=project,
            mode='default',
        )
        compile_elapsed = time.perf_counter() - compile_start

        print("    Compilation completed!")
        print(f"    Compile time: {compile_elapsed:.3f}s")
        return llvm_mod

    return None


def test_first_vit_linear(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first Linear distribution in ViT (attention q projection style input)."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    model_path = VIT_CONFIGS[model_name]["model_path"]
    seq_len = 197
    # Embedding/block token-like distribution: zero-mean, moderate variance.
    calib_inputs = torch.randn(sample_batch_size, seq_len, n_embd) * 1.25
    test_inputs = torch.randn(test_batch_size, seq_len, n_embd) * 1.40
    module = SingleInputModule(nn.Linear(n_embd, n_embd)).eval()

    # Load HF weights for the first Linear (attention query projection)
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()
    module.mod.weight.data = hf_vit.vit.encoder.layer[0].attention.attention.query.weight.data
    module.mod.bias.data = hf_vit.vit.encoder.layer[0].attention.attention.query.bias.data

    print("\n" + "=" * 60)
    print(f"Testing First ViT Linear ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="Linear",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[QLinear],
        project="pynq_vivado_first_linear.prj",
    )


def test_first_vit_layernorm_linear(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first ViT LayerNorm + Linear chain with HF layer-0 weights."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    model_path = VIT_CONFIGS[model_name]["model_path"]
    seq_len = 197

    calib_inputs = torch.randn(sample_batch_size, seq_len, n_embd) * 1.20 + 0.10
    test_inputs = torch.randn(test_batch_size, seq_len, n_embd) * 1.35 + 0.15
    module = FirstLayerNormLinearModule(n_embd).eval()

    # Load HF weights for layernorm_before + attention.query in first block.
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()
    first_block = hf_vit.vit.encoder.layer[0]

    module.norm.weight.data = first_block.layernorm_before.weight.data
    module.norm.bias.data = first_block.layernorm_before.bias.data
    module.norm.eps = first_block.layernorm_before.eps

    module.linear_q.weight.data = first_block.attention.attention.query.weight.data
    module.linear_q.bias.data = first_block.attention.attention.query.bias.data

    print("\n" + "=" * 60)
    print(f"Testing First ViT LayerNorm+Linear ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="LayerNorm+Linear",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[IntLayerNorm, QLinear],
        project="pynq_vivado_first_layernorm_linear.prj",
    )


###############################################################################
# NOTE: ln_qk currently not supported for compilation. For it's multi-head output
###############################################################################
def test_first_vit_ln_qk_matmul(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first ViT LayerNorm + QLinear + KLinear + QK^T chain with HF layer-0 weights."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    n_head = VIT_CONFIGS[model_name]["n_head"]
    model_path = VIT_CONFIGS[model_name]["model_path"]
    seq_len = 197

    calib_inputs = torch.randn(sample_batch_size, seq_len, n_embd) * 1.20 + 0.10
    test_inputs = torch.randn(test_batch_size, seq_len, n_embd) * 1.35 + 0.15
    module = FirstLayerNormQKMatMulModule(n_embd=n_embd, n_head=n_head).eval()

    # Load HF weights for layernorm_before + attention.query/key in first block.
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()
    first_block = hf_vit.vit.encoder.layer[0]

    module.norm.weight.data = first_block.layernorm_before.weight.data
    module.norm.bias.data = first_block.layernorm_before.bias.data
    module.norm.eps = first_block.layernorm_before.eps

    module.linear_q.weight.data = first_block.attention.attention.query.weight.data
    module.linear_q.bias.data = first_block.attention.attention.query.bias.data
    module.linear_k.weight.data = first_block.attention.attention.key.weight.data
    module.linear_k.bias.data = first_block.attention.attention.key.bias.data

    print("\n" + "=" * 60)
    print(f"Testing First ViT LN+QLinear+KLinear+QK^T ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="LN+QLinear+KLinear+QK^T",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[IntLayerNorm, QLinear, QMatMul, QMatMulIsqrtD],
        project="pynq_vivado_first_ln_qk_matmul.prj",
    )


def test_first_vit_softmax(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first Softmax distribution in ViT attention scores."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_head = VIT_CONFIGS[model_name]["n_head"]
    seq_len = 197
    # Attention-score style distribution before softmax.
    # NOTE: n_head pointed to 1 and squeezed
    calib_inputs = torch.randn(sample_batch_size, seq_len, seq_len) * 1.15
    test_inputs = torch.randn(test_batch_size, seq_len, seq_len) * 1.30
    module = SingleInputModule(nn.Softmax(dim=-1)).eval()

    print("\n" + "=" * 60)
    print(f"Testing First ViT Softmax ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="Softmax",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[IntSoftmax],
        project="pynq_vivado_first_softmax.prj",
    )


def test_first_vit_layernorm(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first LayerNorm distribution in ViT block input."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    model_path = VIT_CONFIGS[model_name]["model_path"]
    seq_len = 197
    calib_inputs = torch.randn(sample_batch_size, seq_len, n_embd) * 1.20 + 0.10
    test_inputs = torch.randn(test_batch_size, seq_len, n_embd) * 1.35 + 0.15
    module = SingleInputModule(nn.LayerNorm(n_embd)).eval()

    # Load HF weights for the first LayerNorm (layernorm_before in first block)
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()
    module.mod.weight.data = hf_vit.vit.encoder.layer[0].layernorm_before.weight.data
    module.mod.bias.data = hf_vit.vit.encoder.layer[0].layernorm_before.bias.data
    module.mod.eps = hf_vit.vit.encoder.layer[0].layernorm_before.eps

    print("\n" + "=" * 60)
    print(f"Testing First ViT LayerNorm ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="LayerNorm",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[IntLayerNorm],
        project="pynq_vivado_first_layernorm.prj",
    )


def test_first_vit_gelu(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first GELU distribution in ViT FFN intermediate activation."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    seq_len = 197
    ffn_hidden_dim = n_embd * 4
    calib_inputs = torch.randn(sample_batch_size, seq_len, ffn_hidden_dim) * 1.00
    test_inputs = torch.randn(test_batch_size, seq_len, ffn_hidden_dim) * 1.20
    module = SingleInputModule(nn.GELU()).eval()

    print("\n" + "=" * 60)
    print(f"Testing First ViT GELU ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="GELU",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[IntGELU],
        project="pynq_vivado_first_gelu.prj",
    )


def test_first_vit_matmul(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first MatMul distribution in ViT attention score computation (q @ k^T)."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    n_head = VIT_CONFIGS[model_name]["n_head"]
    seq_len = 197
    head_dim = n_embd // n_head

    # MatMulIsqrtD in ViT: (B, H, S, D) @ (B, H, D, S)
    q_calib = torch.randn(sample_batch_size, n_head, seq_len, head_dim) * 0.90
    k_t_calib = torch.randn(sample_batch_size, n_head, head_dim, seq_len) * 0.90
    q_test = torch.randn(test_batch_size, n_head, seq_len, head_dim) * 1.05
    k_t_test = torch.randn(test_batch_size, n_head, head_dim, seq_len) * 1.05

    module = TwoInputsModule(MatMulIsqrtD(head_dim)).eval()

    print("\n" + "=" * 60)
    print(f"Testing First ViT MatMul ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="MatMul",
        module=module,
        calib_inputs=[q_calib, k_t_calib],
        test_inputs=[q_test, k_t_test],
        run_compile=run_compile,
        leaf_modules=[QMatMul, QMatMulIsqrtD],
        project="pynq_vivado_first_matmul.prj",
    )

def test_first_vit_add(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first Add distribution in ViT shortcut computation (x + y)."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    n_head = VIT_CONFIGS[model_name]["n_head"]
    seq_len = 197
    head_dim = n_embd // n_head

    # Add in ViT: (B, S, D) + (B, S, D)
    x_calib = torch.randn(sample_batch_size, seq_len, head_dim) * 0.90
    y_calib = torch.randn(sample_batch_size, seq_len, head_dim) * 0.90
    x_test = torch.randn(test_batch_size, seq_len, head_dim) * 1.05
    y_test = torch.randn(test_batch_size, seq_len, head_dim) * 1.05

    module = TwoInputsModule(Add()).eval()

    print("\n" + "=" * 60)
    print(f"Testing First ViT Add ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="Add",
        module=module,
        calib_inputs=[x_calib, y_calib],
        test_inputs=[x_test, y_test],
        run_compile=run_compile,
        leaf_modules=[QAdd],
        project="pynq_vivado_first_add.prj",
    )


def test_first_vit_attention(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first Attention module in ViT block with HF layer-0 weights."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    n_head = VIT_CONFIGS[model_name]["n_head"]
    model_path = VIT_CONFIGS[model_name]["model_path"]
    seq_len = 197

    calib_inputs = torch.randn(sample_batch_size, seq_len, n_embd) * 1.20 + 0.05
    test_inputs = torch.randn(test_batch_size, seq_len, n_embd) * 1.35 + 0.08

    blk = ViTBlock(n_embd=n_embd, num_heads=n_head, ffn_hidden_dim=n_embd * 4).eval()
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()
    load_vit_block_from_hf(blk, hf_vit, layer_idx=0)
    module = SingleInputModule(blk.attention).eval()

    print("\n" + "=" * 60)
    print(f"Testing First ViT Attention ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="Attention",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[QLinear, IntSoftmax, QMatMul, QMatMulIsqrtD],
        project="pynq_vivado_first_attention.prj",
    )


def test_first_vit_ffn(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    run_compile: bool = False,
):
    """Test first FFN module in ViT block with HF layer-0 weights."""
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    n_embd = VIT_CONFIGS[model_name]["n_embd"]
    n_head = VIT_CONFIGS[model_name]["n_head"]
    model_path = VIT_CONFIGS[model_name]["model_path"]
    seq_len = 197

    calib_inputs = torch.randn(sample_batch_size, seq_len, n_embd) * 1.00
    test_inputs = torch.randn(test_batch_size, seq_len, n_embd) * 1.20

    blk = ViTBlock(n_embd=n_embd, num_heads=n_head, ffn_hidden_dim=n_embd * 4).eval()
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()
    load_vit_block_from_hf(blk, hf_vit, layer_idx=0)
    module = SingleInputModule(blk.ffn).eval()

    print("\n" + "=" * 60)
    print(f"Testing First ViT FFN ({model_name})")
    print("=" * 60)
    return _run_single_op_quant_test(
        op_name="FFN",
        module=module,
        calib_inputs=calib_inputs,
        test_inputs=test_inputs,
        run_compile=run_compile,
        leaf_modules=[QLinear, IntGELU],
        project="pynq_vivado_first_ffn.prj",
    )

# ==============================================================================
# Test Functions
# ==============================================================================

def test_calibrate_vit_block(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    dataset_path: str = "/root/data/dataset/imagenet-1k-test",
    run_compile: bool = False,
):
    """Test quantization calibration for a single ViT block using HF weights."""
    print("=" * 60)
    print("Testing ViT Block Calibration")
    print("=" * 60)

    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")

    config = VIT_CONFIGS[model_name]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    model_path = config["model_path"]

    # Fixed parameters
    n_channels = 3
    patch_size = (16, 16)
    img_size = (224, 224)

    print(f"\n[Config] n_embd={n_embd}, n_head={n_head}")
    print(f"[Config] model_path={model_path}")
    print(f"[Config] dataset_path={dataset_path}")

    # Load dataset
    print("\n[1] Loading ImageNet test data...")
    example_val_data = get_imagenet_test_data(dataset_path, sample_batch_size, img_size[0], is_reverse_sample=True)
    test_val_data = get_imagenet_test_data(dataset_path, test_batch_size, img_size[0])

    example_images = torch.concat([x[0] for x in example_val_data])
    test_images = torch.concat([x[0] for x in test_val_data])

    print(f"    Calibration samples: {example_images.shape}")
    print(f"    Test samples: {test_images.shape}")

    # Load HuggingFace model and embeddings
    print("\n[2] Loading HuggingFace model...")
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()

    print("\n[3] Preparing embedding inputs...")
    with torch.no_grad():
        example_inputs = hf_vit.vit.embeddings(example_images)
        test_inputs = hf_vit.vit.embeddings(test_images)

    # save the embedded input for backend test
    test_inputs[0].numpy().tofile("in_1_197_192_float32.bin")

    # Build block and load HF weights
    print("\n[4] Loading first block weights...")
    blk = ViTBlock(n_embd=n_embd, num_heads=n_head, ffn_hidden_dim=n_embd * 4).eval()
    load_vit_block_from_hf(blk, hf_vit, layer_idx=0)

    # Quantize block
    print("\n[5] Replacing modules with quantized versions...")
    quant_config = get_vit_optimized_config()
    if run_compile:
        set_extra_compile_param_for_config(quant_config)
    set_extra_compile_param_for_config(quant_config)
    qblk = replace_module_with_quantized(blk, config=quant_config)

    # Calibrate with embedding outputs
    print("\n[6] Calibrating...")
    calibrator = Calibrator(qblk, example_inputs)
    calibrator.calibrate()

    if not run_compile:
        calibrator.enable_fakequant()
        calibrator.enable_lut_inference()
        with torch.no_grad():
            golden = blk(test_inputs)
            res_fake = qblk(test_inputs)

        mean_diff = torch.mean(torch.abs(golden - res_fake)).item()
        max_diff = torch.max(torch.abs(golden - res_fake)).item()

        print("\n" + "-" * 40)
        print("Results (float vs fakequant):")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Max diff: {max_diff:.6f}")
        # Use the new visualization comparison function
        compare_result = compare_tensors_with_plot(
            golden, 
            res_fake,
            name="ViT Block (Float vs FakeQuant)",
            tolerance_rtol=1e-5,
            tolerance_atol=1e-6,
            enable_plot=True,
            plot_filename="vit_block_comparison.png"
        )
    else:
        # Compile mode
        batch = 1
        print("\n[7] Compiling first block with allo...")
        compile_start = time.perf_counter()
        llvm_mod = allo.frontend.from_pytorch_vivado(
            qblk,
            example_inputs=[example_inputs[:batch]],
            leaf_modules=[QLinear, IntLayerNorm, IntSoftmax, IntGELU, QAdd, QMatMul, QMatMulIsqrtD],
            quant_config=quant_config,
            verbose=False,
            project='pynq_vivado_block.prj',
            mode='default',
        )
        compile_elapsed = time.perf_counter() - compile_start
        print(f"    Compile time: {compile_elapsed:.3f}s")

        return llvm_mod


def test_calibrate_vit(
    model_name: str = "deit-tiny",
    sample_batch_size: int = 32,
    test_batch_size: int = 200,
    dataset_path: str = "/root/data/dataset/imagenet-1k-test",
    run_compile: bool = False,
):
    """
    Test quantization calibration for a complete ViT model.
    
    Args:
        model_name: One of "deit-tiny", "deit-small", "deit-base", "vit-base"
        sample_batch_size: Batch size for calibration samples
        test_batch_size: Batch size for testing
        dataset_path: Path to ImageNet test dataset
        run_compile: If True, compile the model with allo instead of running inference
    """
    print("\n" + "=" * 60)
    print(f"Testing ViT Model Calibration: {model_name}")
    print("=" * 60)

    # Get model configuration
    if model_name not in VIT_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VIT_CONFIGS.keys())}")
    
    config = VIT_CONFIGS[model_name]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layers = config["n_layers"]
    model_path = config["model_path"]
    
    # Fixed parameters
    n_channels = 3
    patch_size = (16, 16)
    img_size = (224, 224)
    n_cls = 1000

    print(f"\n[Config] n_embd={n_embd}, n_head={n_head}, n_layers={n_layers}")
    print(f"[Config] model_path={model_path}")
    print(f"[Config] dataset_path={dataset_path}")

    # Load dataset
    print("\n[1] Loading ImageNet test data...")
    example_val_data = get_imagenet_test_data(dataset_path, sample_batch_size, img_size[0], is_reverse_sample=True)
    test_val_data = get_imagenet_test_data(dataset_path, test_batch_size, img_size[0])
    
    example_inputs = torch.concat([x[0] for x in example_val_data])
    test_inputs = torch.concat([x[0] for x in test_val_data])
    test_labels = torch.tensor([x[1] for x in test_val_data], dtype=torch.int)
    
    print(f"    Calibration samples: {example_inputs.shape}")
    print(f"    Test samples: {test_inputs.shape}")

    # Create model
    print("\n[2] Creating ViT model...")
    vit = ViTImgCls(n_embd, n_head, n_layers, n_channels, patch_size, img_size, n_cls).eval()

    # Load HuggingFace weights
    print("\n[3] Loading HuggingFace model weights...")
    from transformers import ViTForImageClassification
    hf_vit = ViTForImageClassification.from_pretrained(model_path).eval()
    replace_vit_with_hf_vit(vit, hf_vit)

    # Quantize model
    print("\n[4] Replacing modules with quantized versions...")
    # quant_config = get_default_config() 
    quant_config = get_vit_optimized_config()
    if run_compile:
        set_extra_compile_param_for_config(quant_config)
    set_extra_compile_param_for_config(quant_config)
    vit = replace_module_with_quantized(vit, config=quant_config)

    # Calibrate
    print("\n[5] Calibrating...")
    calibrator = Calibrator(vit, example_inputs)
    calibrator.calibrate()

    if run_compile:
        # Compile mode
        batch = 1
        print("\n[6] Compiling with allo...")
        compile_start = time.perf_counter()
        llvm_mod = allo.frontend.from_pytorch_vivado(
            vit,
            example_inputs=[example_inputs[:batch]],
            leaf_modules=[ViTGetFirstToken, ViTTokenExpand, QLinear, QConv2d, IntLayerNorm, IntSoftmax, IntGELU, QAdd, QMatMul, QMatMulIsqrtD],
            quant_config=quant_config,
            verbose=False,
            project='pynq_vivado.prj',
            mode='default',
        )
        compile_elapsed = time.perf_counter() - compile_start
        print("    Compilation completed!")
        print(f"    Compile time: {compile_elapsed:.3f}s")
        return llvm_mod
    else:
        calibrator.enable_fakequant()
        calibrator.enable_lut_inference()
        # Inference mode
        print("\n[6] Running inference test...")
        total = test_batch_size
        top1_match = 0
        top1_acc = 0
        top1_acc_ref = 0
        mean_diff = 0
        max_diff = torch.tensor(0.0)
        step = 20
        
        with torch.no_grad():
            for i in range(0, total, step):
                inp = test_inputs[i:min(i+step, total)]
                lbl = test_labels[i:min(i+step, total)]
                golden = hf_vit(inp).logits
                res = vit(inp)
                
                top1_match += torch.sum(golden.argmax(-1) == res.argmax(-1))
                top1_acc += torch.sum(lbl == res.argmax(-1))
                top1_acc_ref += torch.sum(lbl == golden.argmax(-1))
                mean_diff += torch.mean(torch.abs(golden - res))
                max_diff = torch.max(torch.abs(golden - res).max(), max_diff)
        
        mean_diff = mean_diff * step / total

        # Print results
        print("\n" + "-" * 40)
        print("Results:")
        print(f"    Top1 Match (quant vs ref): {top1_match}/{total} ({100*top1_match/total:.1f}%)")
        print(f"    Top1 Accuracy (quant):     {top1_acc}/{total} ({100*top1_acc/total:.1f}%)")
        print(f"    Top1 Accuracy (ref):       {top1_acc_ref}/{total} ({100*top1_acc_ref/total:.1f}%)")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Max diff: {max_diff:.6f}")

        return {
            "top1_match": top1_match.item(),
            "top1_acc": top1_acc.item(),
            "top1_acc_ref": top1_acc_ref.item(),
            "mean_diff": mean_diff.item(),
            "max_diff": max_diff.item(),
            "total": total,
        }


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ViT INT8 Quantization Test")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deit-tiny",
        choices=list(VIT_CONFIGS.keys()),
        help="ViT model variant to test"
    )
    parser.add_argument(
        "--sample-batch", 
        type=int, 
        default=32,
        help="Batch size for calibration samples"
    )
    parser.add_argument(
        "--test-batch", 
        type=int, 
        default=200,
        help="Batch size for test samples"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="/root/data/dataset/imagenet-1k-test",
        help="Path to ImageNet test dataset"
    )
    parser.add_argument(
        "--compile", 
        action="store_true",
        help="Compile model with allo instead of running inference"
    )
    parser.add_argument(
        "--test-block", 
        action="store_true",
        help="Only test a single ViT block (no model loading)"
    )
    parser.add_argument(
        "--test-op",
        type=str,
        default=None,
        choices=["linear", "ln_linear", "ln_qk", "softmax", "layernorm", "gelu", "matmul", "add", "attention", "ffn"],
        help="Test first occurrence of a single ViT operator type"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("    ViT INT8 Quantization Test (Refactored)")
    print("=" * 60)
    
    if args.test_op is not None:
        if args.test_op == "linear":
            test_first_vit_linear(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "ln_linear":
            test_first_vit_layernorm_linear(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "ln_qk":
            test_first_vit_ln_qk_matmul(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "softmax":
            test_first_vit_softmax(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "layernorm":
            test_first_vit_layernorm(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "gelu":
            test_first_vit_gelu(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "matmul":
            test_first_vit_matmul(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "add":
            test_first_vit_add(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "attention":
            test_first_vit_attention(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
        elif args.test_op == "ffn":
            test_first_vit_ffn(
                model_name=args.model,
                sample_batch_size=args.sample_batch,
                test_batch_size=args.test_batch,
                run_compile=args.compile,
            )
    elif args.test_block:
        test_calibrate_vit_block(
            model_name=args.model,
            sample_batch_size=args.sample_batch,
            test_batch_size=args.test_batch,
            dataset_path=args.dataset,
            run_compile=args.compile,
        )
    else:
        test_calibrate_vit(
            model_name=args.model,
            sample_batch_size=args.sample_batch,
            test_batch_size=args.test_batch,
            dataset_path=args.dataset,
            run_compile=args.compile,
        )


if __name__ == "__main__":
    main()
