"""
Allo Quantization Module

This module provides quantization utilities for neural network models.
"""

from .quant_utils import (
    max_min_quantize_params,
    mean_std_quantize_params,
    linear_quantize,
    symmetric_linear_quantize,
    asymmetric_linear_quantize,
)

from .quant_modules import (
    QuantizableModule,
    QLinear,
    QConv2d,
    QFFN,
    IntGELU,
    IntSoftmax,
    IntLayerNorm,
    QAct,
    QAdd,
    QMatMul,
    QMatMulIsqrtD,
)

from .quant_config import (
    LayerQuantConfig,
    QuantConfig,
    Calibrator,
    replace_module_with_quantized,
    get_default_config,
    get_per_token_config,
    get_asymmetric_config,
    get_vit_optimized_config,
)

__all__ = [
    # Utils
    'max_min_quantize_params',
    'mean_std_quantize_params',
    'linear_quantize',
    'symmetric_linear_quantize',
    'asymmetric_linear_quantize',
    # Modules
    'QuantizableModule',
    'QLinear',
    'QConv2d',
    'QFFN',
    'IntGELU',
    'IntSoftmax',
    'IntLayerNorm',
    'QAct',
    'QAdd',
    'QMatMul',
    'QMatMulIsqrtD',
    # Config
    'LayerQuantConfig',
    'QuantConfig',
    'Calibrator',
    'replace_module_with_quantized',
    'get_default_config',
    'get_per_token_config',
    'get_asymmetric_config',
    'get_vit_optimized_config',
]
