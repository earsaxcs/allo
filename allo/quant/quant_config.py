"""
Quantization Configuration System

This module provides a flexible configuration system for quantizing neural network models.
It allows users to specify quantization parameters (sym/asym, per-token/per-tensor, bitwidth)
for different layer types or specific layers by name.

Example Usage:
    # Create a default config
    config = QuantConfig()
    
    # Customize specific layer types
    config.set_layer_type_config(nn.Linear, act_quant_mode="asym", wgt_per_channel=True)
    config.set_layer_type_config(nn.Softmax, act_per_token=True)
    
    # Customize specific layers by name pattern
    config.set_layer_name_config("attention.linear_q", act_quant_mode="sym")
    config.set_layer_name_config("ffn.*", act_bit=8)  # supports regex
    
    # Replace model with quantized version
    qmodel = replace_module_with_quantized(model, config)
"""

import copy
import re
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, List, Union

# Import quantization modules
from .quant_modules import (
    QLinear, QConv2d, IntGELU, IntSoftmax, IntLayerNorm,
    QAdd, QMatMul, QMatMulIsqrtD, QAct, QuantizableModule
)

# Try to import custom ops
try:
    from ..ops.vit import Add, MatMul, MatMulIsqrtD
except ImportError:
    Add = None
    MatMul = None
    MatMulIsqrtD = None

# GLOBAL SETTINGS
DEFAULT_ACT_BIT = 8
DEFAULT_WEIGHT_BIT = 8
DEFAULT_BIAS_BIT = 8

# Global configuration for quantization
# SCALE_FIXED_BITS: Number of bits used to represent scale coefficient in [0.5, 1.0)
# Default: 17 bits (coe ∈ [65536, 131071] represents [0.5, 1.0))
# This value should match kExpectedAlloFixedBits in LowerAlloQuantToVivado.cpp
#
# Configuration Flow:
# 1. Set SCALE_FIXED_BITS here (default: 17)
# 2. float_to_fixed_point() uses this to convert scales: coe = value * 2^SCALE_FIXED_BITS
# 3. In LowerAlloQuantToVivado.cpp, kExpectedAlloFixedBits should match this value
# 4. The lowering pass validates and packs coe according to scale_coe_mode (Tail/Full)
#
# Example:
#   SCALE_FIXED_BITS = 17 means:
#   - scale 0.5 → coe = 0.5 * 2^17 = 65536 = 0b10000000000000000
#   - scale 0.625 → coe = 0.625 * 2^17 = 81920 = 0b10100000000000000
#   - Tail mode: stores low 16 bits (variant part)
#   - Full mode: stores high 16 bits (including leading 1)
DEFAULT_SCALE_FIXED_BITS = 17

DEFAULT_SEQ_LEN = 197  # Default sequence length for models like ViT
@dataclass
class LayerQuantConfig:
    """Configuration for a single layer's quantization parameters."""
    # Common parameters
    act_bit: int = DEFAULT_ACT_BIT
    act_quant_mode: str = "sym"  # "sym" or "asym"
    # Legacy: per-token (True) or per-tensor (False). Used as fallback.
    act_per_token: bool = False

    # New: split per-token behavior for input/output activations.
    # If None, falls back to act_per_token.
    input_act_per_token: Optional[bool] = None
    output_act_per_token: Optional[bool] = None
    
    # Linear/Conv specific
    weight_bit: int = DEFAULT_WEIGHT_BIT
    bias_bit: int = DEFAULT_BIAS_BIT
    wgt_per_channel: bool = False
    
    # Softmax specific
    in_act_bit: int = 8
    softmax_act_bit: int = 16
    out_act_bit: int = 8
    act_per_head: bool = False
    
    # LayerNorm specific
    # Legacy: per hidden-dim (True) or per-tensor (False). Used as fallback.
    act_per_channel: bool = False
    # New: split LayerNorm input/output behavior.
    # - input_act_per_token: reuse the common field above
    # - output_act_per_channel: if None, falls back to act_per_channel
    output_act_per_channel: Optional[bool] = None
    int_cal_mode: str = "I-ViT"
    
    # MatMul specific (inherits act_* params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'act_bit': self.act_bit,
            'act_quant_mode': self.act_quant_mode,
            'act_per_token': self.act_per_token,
            'input_act_per_token': self.input_act_per_token,
            'output_act_per_token': self.output_act_per_token,
            'weight_bit': self.weight_bit,
            'bias_bit': self.bias_bit,
            'wgt_per_channel': self.wgt_per_channel,
            'in_act_bit': self.in_act_bit,
            'softmax_act_bit': self.softmax_act_bit,
            'out_act_bit': self.out_act_bit,
            'act_per_head': self.act_per_head,
            'act_per_channel': self.act_per_channel,
            'output_act_per_channel': self.output_act_per_channel,
            'int_cal_mode': self.int_cal_mode,
        }
    
    def copy(self) -> 'LayerQuantConfig':
        """Create a copy of this config."""
        return LayerQuantConfig(**self.to_dict())


class QuantConfig:
    """
    Main configuration class for model quantization.
    
    Supports:
    - Default configuration for all layers
    - Per-layer-type configuration (e.g., all nn.Linear layers)
    - Per-layer-name configuration (supports regex patterns)
    - Priority: layer_name > layer_type > default
    """
    # Global fixed point settings
    scale_fixed_bits: int = DEFAULT_SCALE_FIXED_BITS
    # Global Sequence length
    seq_len: int = DEFAULT_SEQ_LEN
    
    # Mapping from original module types to quantized module classes
    QUANT_MODULE_MAP: Dict[Type[nn.Module], Type[nn.Module]] = {
        nn.Linear: QLinear,
        # nn.Conv2d: QConv2d,
        nn.GELU: IntGELU,
        nn.Softmax: IntSoftmax,
        nn.LayerNorm: IntLayerNorm,
        # Note: Add, MatMul, MatMulIsqrtD are added in __init__ if available
    }
    
    def __init__(self, default_config: Optional[LayerQuantConfig] = None):
        """
        Initialize QuantConfig.
        
        Args:
            default_config: Default configuration for all layers. If None, uses LayerQuantConfig defaults.
        """
        self.default_config = default_config or LayerQuantConfig()
        self.layer_type_configs: Dict[Type[nn.Module], LayerQuantConfig] = {}
        self.layer_name_configs: Dict[str, LayerQuantConfig] = {}  # name pattern -> config
        self.skip_layers: List[str] = []  # Layer name patterns to skip quantization
        
        # Add custom ops to mapping if available
        if Add is not None:
            self.QUANT_MODULE_MAP[Add] = QAdd
        if MatMul is not None:
            self.QUANT_MODULE_MAP[MatMul] = QMatMul
        if MatMulIsqrtD is not None:
            self.QUANT_MODULE_MAP[MatMulIsqrtD] = QMatMulIsqrtD
    
    def set_default_config(self, **kwargs) -> 'QuantConfig':
        """
        Set default quantization parameters.
        
        Args:
            **kwargs: Any LayerQuantConfig parameters
        
        Returns:
            self for method chaining
        """
        for key, value in kwargs.items():
            if hasattr(self.default_config, key):
                setattr(self.default_config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self
    
    def set_layer_type_config(self, layer_type: Type[nn.Module], **kwargs) -> 'QuantConfig':
        """
        Set quantization parameters for a specific layer type.
        
        Args:
            layer_type: The nn.Module type (e.g., nn.Linear, nn.Conv2d)
            **kwargs: Any LayerQuantConfig parameters
        
        Returns:
            self for method chaining
        """
        if layer_type not in self.layer_type_configs:
            self.layer_type_configs[layer_type] = self.default_config.copy()
        
        config = self.layer_type_configs[layer_type]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self
    
    def set_layer_name_config(self, name_pattern: str, **kwargs) -> 'QuantConfig':
        """
        Set quantization parameters for layers matching a name pattern.
        
        Args:
            name_pattern: Layer name or regex pattern (e.g., "fc1", "attention.*", ".*norm.*")
            **kwargs: Any LayerQuantConfig parameters
        
        Returns:
            self for method chaining
        """
        if name_pattern not in self.layer_name_configs:
            self.layer_name_configs[name_pattern] = self.default_config.copy()
        
        config = self.layer_name_configs[name_pattern]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self
    
    def skip_layer(self, name_pattern: str) -> 'QuantConfig':
        """
        Skip quantization for layers matching the name pattern.
        
        Args:
            name_pattern: Layer name or regex pattern
        
        Returns:
            self for method chaining
        """
        self.skip_layers.append(name_pattern)
        return self
    
    def should_skip(self, layer_name: str) -> bool:
        """Check if a layer should be skipped."""
        for pattern in self.skip_layers:
            if re.fullmatch(pattern, layer_name) or pattern in layer_name:
                return True
        return False
    
    def get_config_for_layer(self, layer_name: str, layer_type: Type[nn.Module]) -> LayerQuantConfig:
        """
        Get the quantization config for a specific layer.
        Priority: layer_name > layer_type > default
        
        Args:
            layer_name: Full name of the layer (e.g., "encoder.layer.0.attention.self.query")
            layer_type: The type of the layer (e.g., nn.Linear)
        
        Returns:
            LayerQuantConfig for the layer
        """
        # Start with default config
        config = self.default_config.copy()
        
        # Apply layer type config if exists
        if layer_type in self.layer_type_configs:
            type_config = self.layer_type_configs[layer_type]
            for key, value in type_config.to_dict().items():
                setattr(config, key, value)
        
        # Apply layer name config if matches (highest priority)
        for pattern, name_config in self.layer_name_configs.items():
            if re.fullmatch(pattern, layer_name) or pattern in layer_name:
                for key, value in name_config.to_dict().items():
                    setattr(config, key, value)
                break  # First match wins
        
        return config
    
    def get_quant_module_class(self, layer_type: Type[nn.Module]) -> Optional[Type[nn.Module]]:
        """Get the quantized module class for a given layer type."""
        return self.QUANT_MODULE_MAP.get(layer_type)
    
    def __repr__(self) -> str:
        lines = ["QuantConfig:"]
        lines.append(f"  Default: {self.default_config}")
        if self.layer_type_configs:
            lines.append("  Layer Type Configs:")
            for lt, cfg in self.layer_type_configs.items():
                lines.append(f"    {lt.__name__}: {cfg}")
        if self.layer_name_configs:
            lines.append("  Layer Name Configs:")
            for name, cfg in self.layer_name_configs.items():
                lines.append(f"    '{name}': {cfg}")
        if self.skip_layers:
            lines.append(f"  Skip Patterns: {self.skip_layers}")
        return "\n".join(lines)


# ============== Preset Configurations ==============

def get_default_config() -> QuantConfig:
    """Get default quantization config (symmetric, per-tensor)."""
    return QuantConfig()


def get_per_token_config() -> QuantConfig:
    """Get per-token quantization config for activations."""
    config = QuantConfig()
    config.set_default_config(act_per_token=True)
    config.set_layer_type_config(nn.Linear, wgt_per_channel=False)
    config.set_layer_type_config(nn.Conv2d, wgt_per_channel=False)
    config.set_layer_type_config(nn.LayerNorm, act_per_channel=True)
    return config


def get_asymmetric_config() -> QuantConfig:
    """Get asymmetric quantization config."""
    config = QuantConfig()
    config.set_default_config(act_quant_mode="asym")
    return config


def get_vit_optimized_config() -> QuantConfig:
    """
    Get optimized quantization config for Vision Transformer.
    - Per-channel for Linear weights
    - Per-token for attention activations
    - Symmetric quantization
    """
    config = QuantConfig()
    
    # Linear layers: per-channel weights
    config.set_layer_type_config(nn.Linear, wgt_per_channel=False, act_per_token=True)
    config.set_layer_name_config("attention.linear_q", wgt_per_channel=False, input_act_per_token=False, output_act_per_token=True)
    config.set_layer_name_config("ffn.fc1", wgt_per_channel=False, input_act_per_token=False, output_act_per_token=True)
    config.set_layer_name_config("attention.linear_k", wgt_per_channel=False, input_act_per_token=False, output_act_per_token=False)
    config.set_layer_name_config("attention.linear_v", wgt_per_channel=False, input_act_per_token=False, output_act_per_token=False)
    config.set_layer_name_config("classifier.dense", wgt_per_channel=False, input_act_per_token=False, output_act_per_token=False)
    
    # Conv2d: per-channel weights
    config.set_layer_type_config(nn.Conv2d, wgt_per_channel=False)
    
    # Softmax: per-token for attention scores
    config.set_layer_type_config(nn.Softmax, act_per_token=True)
    
    # GELU: per-token
    config.set_layer_type_config(nn.GELU, act_per_token=True)
    
    # LayerNorm: per-channel (per hidden dim)
    config.set_layer_type_config(nn.LayerNorm, input_act_per_token=False, output_act_per_channel=False)
    
    # MatMul: per-token
    if MatMul is not None:
        config.set_layer_type_config(MatMul, act_per_token=True)
    if MatMulIsqrtD is not None:
        config.set_layer_type_config(MatMulIsqrtD, act_per_token=True)
    
    # Add: per-token
    if Add is not None:
        config.set_layer_type_config(Add, act_per_token=True)
    
    return config

def set_extra_compile_param_for_config(quant_config: QuantConfig) -> None:
    quant_config.set_layer_type_config(nn.LayerNorm, int_cal_mode="Vivado-PYNQ")

# ============== Model Replacement Function ==============

def _create_quant_module(module: nn.Module, config: LayerQuantConfig, quant_cls: Type[nn.Module]) -> nn.Module:
    """
    Create a quantized module from the original module with the given config.
    
    Args:
        module: Original nn.Module
        config: LayerQuantConfig for this layer
        quant_cls: The quantized module class to instantiate
    
    Returns:
        Quantized module
    """
    cfg = config.to_dict()
    
    if quant_cls == QLinear:
        return QLinear.struct_module(
            module,
            weight_bit=cfg['weight_bit'],
            bias_bit=cfg['bias_bit'],
            act_bit=cfg['act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            act_per_token=cfg['act_per_token'],
            input_act_per_token=cfg.get('input_act_per_token', None),
            output_act_per_token=cfg.get('output_act_per_token', None),
            wgt_per_channel=cfg['wgt_per_channel'],
        )
    
    elif quant_cls == QConv2d:
        return QConv2d.struct_module(
            module,
            weight_bit=cfg['weight_bit'],
            bias_bit=cfg['bias_bit'],
            act_bit=cfg['act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            wgt_per_channel=cfg['wgt_per_channel'],
        )
    
    elif quant_cls == IntGELU:
        return IntGELU.struct_module(
            module,
            act_bit=cfg['act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            act_per_token=cfg['act_per_token'],
        )
    
    elif quant_cls == IntSoftmax:
        return IntSoftmax.struct_module(
            module,
            in_act_bit=cfg['in_act_bit'],
            softmax_act_bit=cfg['softmax_act_bit'],
            out_act_bit=cfg['out_act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            act_per_token=cfg['act_per_token'],
            act_per_head=cfg['act_per_head'],
        )
    
    elif quant_cls == IntLayerNorm:
        return IntLayerNorm.struct_module(
            module,
            in_act_bit=cfg['act_bit'],
            out_act_bit=cfg['act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            act_per_channel=cfg['act_per_channel'],
            input_act_per_token=bool(cfg.get('input_act_per_token', False)) if cfg.get('input_act_per_token', None) is not None else False,
            output_act_per_channel=cfg.get('output_act_per_channel', None),
            int_cal_mode=cfg['int_cal_mode'],
        )
    
    elif quant_cls == QAdd:
        return QAdd.struct_module(
            module,
            act_bit=cfg['act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            act_per_token=cfg['act_per_token'],
        )
    
    elif quant_cls == QMatMul:
        return QMatMul.struct_module(
            module,
            act_bit=cfg['act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            act_per_token=cfg['act_per_token'],
        )
    
    elif quant_cls == QMatMulIsqrtD:
        return QMatMulIsqrtD.struct_module(
            module,
            act_bit=cfg['act_bit'],
            act_quant_mode=cfg['act_quant_mode'],
            act_per_token=cfg['act_per_token'],
        )
    
    else:
        raise ValueError(f"Unsupported quantized module class: {quant_cls}")


def replace_module_with_quantized(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
    inplace: bool = False,
    verbose: bool = False,
) -> nn.Module:
    """
    Replace modules in the model with their quantized versions based on config.
    
    Args:
        model: The original model to quantize
        config: QuantConfig specifying quantization parameters. If None, uses get_vit_optimized_config()
        inplace: If True, modify the model in place. If False, create a deep copy.
        verbose: If True, print information about replaced modules.
    
    Returns:
        Model with quantized modules
    """
    if config is None:
        config = get_vit_optimized_config()
    
    if not inplace:
        model = copy.deepcopy(model)
    
    replaced_count = 0
    skipped_count = 0
    
    def _replace_recursive(parent_module: nn.Module, parent_name: str = ""):
        nonlocal replaced_count, skipped_count
        
        for name, child in list(parent_module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            child_type = type(child)
            
            # First, recurse into children
            _replace_recursive(child, full_name)
            
            # Check if this module type should be quantized
            quant_cls = config.get_quant_module_class(child_type)
            
            if quant_cls is not None:
                # Check if should skip
                if config.should_skip(full_name):
                    if verbose:
                        print(f"  Skipped: {full_name} ({child_type.__name__})")
                    skipped_count += 1
                    continue
                
                # Get config for this layer
                layer_config = config.get_config_for_layer(full_name, child_type)
                
                # Create quantized module
                try:
                    qmodule = _create_quant_module(child, layer_config, quant_cls)
                    setattr(parent_module, name, qmodule)
                    replaced_count += 1
                    
                    if verbose:
                        print(f"  Replaced: {full_name} ({child_type.__name__} -> {quant_cls.__name__})")
                        print(f"    Config: mode={layer_config.act_quant_mode}, "
                              f"per_token={layer_config.act_per_token}, "
                              f"wgt_per_channel={layer_config.wgt_per_channel}")
                except Exception as e:
                    print(f"  Warning: Failed to replace {full_name}: {e}")
    
    if verbose:
        print(f"Replacing modules with quantized versions...")
        print(f"Config: {config}")
    
    _replace_recursive(model)
    
    if verbose:
        print(f"\nReplacement complete: {replaced_count} replaced, {skipped_count} skipped")
    
    return model


# ============== Calibrator Class ==============

class Calibrator:
    """
    Calibrator for quantized models.
    
    Handles:
    - Enabling/disabling calibration mode
    - Running calibration with example inputs
    - Enabling/disabling fake quantization mode
    """
    
    def __init__(
        self,
        qmodel: nn.Module,
        example_inputs: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    ):
        """
        Initialize Calibrator.
        
        Args:
            qmodel: Quantized model
            example_inputs: Example inputs for calibration. Can be:
                - A single tensor
                - A list of tensors (for models with multiple inputs)
                - A dict of tensors (for models with named inputs)
        """
        self.qmodel = qmodel
        self.example_inputs = example_inputs
    
    def _get_quantizable_modules(self):
        """Yield all quantizable modules in the model."""
        for name, module in self.qmodel.named_modules():
            if isinstance(module, QuantizableModule):
                yield name, module
    
    def calibrate(self):
        """Run calibration on the model."""
        self.enable_calibrate()
        
        # Handle different input types
        if isinstance(self.example_inputs, dict):
            self.qmodel(**self.example_inputs)
        elif isinstance(self.example_inputs, (list, tuple)):
            self.qmodel(*self.example_inputs)
        else:
            self.qmodel(self.example_inputs)
        
        self.disable_calibrate()
    
    def enable_calibrate(self):
        """Enable calibration mode for all quantizable modules."""
        for name, module in self._get_quantizable_modules():
            module.start_calibrate()
    
    def disable_calibrate(self):
        """Disable calibration mode for all quantizable modules."""
        for name, module in self._get_quantizable_modules():
            module.stop_calibrate()
    
    def enable_fakequant(self):
        """Enable fake quantization mode for all quantizable modules."""
        for name, module in self._get_quantizable_modules():
            module.enable_fakequant()
    
    def disable_fakequant(self):
        """Disable fake quantization mode for all quantizable modules."""
        for name, module in self._get_quantizable_modules():
            module.disable_fakequant()
    
    def get_quant_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all quantization parameters from the model.
        
        Returns:
            Dictionary mapping layer names to their quantization parameters
        """
        params = {}
        for name, module in self._get_quantizable_modules():
            layer_params = {}
            for attr in ['input_scale', 'output_scale', 'weight_scale', 'bias_scale',
                        'input_zero', 'output_zero', 'fused_scale', 'weight_int', 'bias_int']:
                if hasattr(module, attr):
                    val = getattr(module, attr)
                    if val is not None:
                        layer_params[attr] = val.clone() if isinstance(val, torch.Tensor) else val
            params[name] = layer_params
        return params
    
    def print_quant_summary(self):
        """Print a summary of quantization parameters."""
        print("\n" + "=" * 60)
        print("Quantization Summary")
        print("=" * 60)
        
        for name, module in self._get_quantizable_modules():
            print(f"\n{name} ({type(module).__name__}):")
            
            if hasattr(module, 'input_scale') and module.input_scale is not None:
                print(f"  input_scale: {module.input_scale.shape} = {module.input_scale.flatten()[:4].tolist()}...")
            
            if hasattr(module, 'output_scale') and module.output_scale is not None:
                print(f"  output_scale: {module.output_scale.shape} = {module.output_scale.flatten()[:4].tolist()}...")
            
            if hasattr(module, 'weight_scale') and module.weight_scale is not None:
                print(f"  weight_scale: {module.weight_scale.shape} = {module.weight_scale.flatten()[:4].tolist()}...")
            
            if hasattr(module, 'input_zero') and module.input_zero is not None:
                print(f"  input_zero: {module.input_zero.shape} = {module.input_zero.flatten()[:4].tolist()}...")
        
        print("\n" + "=" * 60)


# ============== Exports ==============

__all__ = [
    'LayerQuantConfig',
    'QuantConfig',
    'Calibrator',
    'replace_module_with_quantized',
    'get_default_config',
    'get_per_token_config',
    'get_asymmetric_config',
    'get_vit_optimized_config',
]
