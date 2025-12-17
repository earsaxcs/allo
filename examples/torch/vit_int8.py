"""
ViT INT8 Quantization Test (Refactored)

This module tests INT8 quantization for Vision Transformer (ViT) models.
Uses allo.ops.vit for model structure and loads real HuggingFace models.

For deprecated manual calibration methods, see vit_int8_legacy.py
"""

import allo
import torch
import torch.nn as nn

# Use allo.ops.vit structures to ensure compatibility
from allo.ops.vit import (
    ViTImgCls, ViTBlock, ViTGetFirstToken, ViTTokenExpand,
)

# Import quantization utilities from quant_config (no duplicate definitions!)
from allo.quant.quant_config import (
    replace_module_with_quantized,
    Calibrator,
    get_default_config,
    get_vit_optimized_config,
)
from allo.quant.quant_modules import (
    QLinear, QConv2d, IntGELU, IntSoftmax, IntLayerNorm, QAdd, QMatMul,
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


# ==============================================================================
# Test Functions
# ==============================================================================

def test_calibrate_vit_block():
    """Test quantization calibration for a single ViT block."""
    print("=" * 60)
    print("Testing ViT Block Calibration")
    print("=" * 60)

    n_embd = 384
    n_head = 6
    sample_batch_size = 32
    batch_size = 10

    blk = ViTBlock(n_embd=n_embd, num_heads=n_head, ffn_hidden_dim=n_embd * 4)
    example_inputs = torch.randn(sample_batch_size, 197, n_embd) * 12
    test_inputs = torch.randn(batch_size, 197, n_embd) * 16

    # Use quant_config's replace and calibrator
    quant_config = get_vit_optimized_config()
    qblk = replace_module_with_quantized(blk, config=quant_config)
    calibrator = Calibrator(qblk, example_inputs)
    calibrator.calibrate()
    calibrator.enable_fakequant()

    golden = blk(test_inputs)
    res = qblk(test_inputs)

    mean_diff = torch.mean(torch.abs(golden - res)).item()
    max_diff = torch.max(torch.abs(golden - res)).item()

    print(f"Mean diff: {mean_diff:.6f}")
    print(f"Max diff: {max_diff:.6f}")
    
    return mean_diff, max_diff


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
    vit = replace_module_with_quantized(vit, config=quant_config)

    # Calibrate
    print("\n[5] Calibrating...")
    calibrator = Calibrator(vit, example_inputs)
    calibrator.calibrate()
    calibrator.enable_fakequant()

    if run_compile:
        # Compile mode
        print("\n[6] Compiling with allo...")
        llvm_mod = allo.frontend.from_pytorch_vivado(
            vit,
            example_inputs=[example_inputs[:2]],
            leaf_modules=[ViTGetFirstToken, ViTTokenExpand, QLinear, QConv2d, IntLayerNorm, IntSoftmax, IntGELU, QAdd, QMatMul],
            verbose=False,
        )
        print("    Compilation completed!")
        return llvm_mod
    else:
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
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("    ViT INT8 Quantization Test (Refactored)")
    print("=" * 60)
    
    if args.test_block:
        test_calibrate_vit_block()
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
