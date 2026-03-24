import allo
import torch
import math
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .quant_utils import *
from typing import Any
from ..ops.vit import Add, MatMul, MatMulIsqrtD

# ----- LUT (Lookup Table) Initialization for Vector Operations -----

# Global LUT initialization flag
_g_lut_initialized = False
_g_gelu_lut = None
_g_softmax_ex_lut = None
_g_softmax_loge_lut = None
_g_norm_insqrt_lut = None
_g_gelu_rev_lut = None
_g_softmax_ex_rev_lut = None

def _double_to_fixed(value: float, m: int, n: int, signed_value: bool = True) -> int:
    """Convert double to fixed-point representation."""
    total_bits = (1 if signed_value else 0) + m + n
    if total_bits <= 0 or total_bits > 31:
        return 0
    
    scale = float(1 << n)
    fixed = int(value * scale)
    
    if signed_value:
        max_v = (1 << (total_bits - 1)) - 1
        min_v = -(1 << (total_bits - 1))
        fixed = max(min_v, min(max_v, fixed))
    else:
        max_u = (1 << total_bits) - 1
        fixed = max(0, min(max_u, fixed))
    
    return fixed

def _gelu_func(x: float) -> float:
    """GELU activation function."""
    k = 0.7978845608028654  # sqrt(2/pi)
    return 0.5 * x * (1.0 + math.tanh(k * (x + 0.044715 * x * x * x)))

def _init_vector_luts():
    """Initialize lookup tables for GELU, Softmax, and LayerNorm operations."""
    global _g_lut_initialized, _g_gelu_lut, _g_softmax_ex_lut, _g_softmax_loge_lut, _g_norm_insqrt_lut
    global _g_gelu_rev_lut, _g_softmax_ex_rev_lut
    
    if _g_lut_initialized:
        return
    
    LUT_SIZE = 1024
    
    # GELU LUT: input range [-4, 4]
    gelu_step = 8.0 / LUT_SIZE
    gelu_lut = torch.zeros(LUT_SIZE, dtype=torch.int16)
    for i in range(LUT_SIZE):
        x_gelu = -4.0 + gelu_step * i
        x_gelu = max(-4.0, min(4.0, x_gelu))
        gelu_lut[i] = _double_to_fixed(_gelu_func(x_gelu), 2, 13, True)
    
    # Softmax exp LUT: input range [-8, 0]
    ex_step = 8.0 / LUT_SIZE
    ex_lut = torch.zeros(LUT_SIZE, dtype=torch.uint16)
    for i in range(LUT_SIZE):
        x_ex = -8.0 + ex_step * i
        x_ex = max(-8.0, min(0.0, x_ex))
        ex_lut[i] = _double_to_fixed(math.exp(x_ex), 0, 16, False)
    
    # Softmax log LUT: input range [0, 256]
    loge_step = 256.0 / LUT_SIZE
    loge_lut = torch.zeros(LUT_SIZE, dtype=torch.int16)
    for i in range(LUT_SIZE):
        x_loge = 0.0 + loge_step * i
        x_loge = max(1.0, min(256.0, x_loge))
        loge_lut[i] = _double_to_fixed(math.log(x_loge), 3, 12, True)
    
    # LayerNorm inverse square root LUT: input range [0, 65536] # [0, 8192]
    LAYERNORM_MAX = 8192
    insqrt_step = LAYERNORM_MAX / LUT_SIZE
    insqrt_lut = torch.zeros(LUT_SIZE, dtype=torch.uint16)
    for i in range(LUT_SIZE):
        x_insqrt = 0.0 + insqrt_step * i
        x_insqrt = max(1.1, min(LAYERNORM_MAX, x_insqrt))
        insqrt_lut[i] = _double_to_fixed(1.0 / math.sqrt(x_insqrt), 0, 16, False)
    
    _g_gelu_lut = gelu_lut
    _g_softmax_ex_lut = ex_lut
    _g_softmax_loge_lut = loge_lut
    _g_norm_insqrt_lut = insqrt_lut
    _g_softmax_ex_rev_lut = torch.roll(ex_lut, shifts=1, dims=0) # 一开始是torch.roll(ex_lut, shifts=-1, dims=0)，错误
    _g_gelu_rev_lut = torch.roll(gelu_lut, shifts=LUT_SIZE // 2, dims=0)
    _g_lut_initialized = True


def _clip_i16(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=(-32768 + (1 << 5)), max=32767).to(torch.int32)


def _extract_bits_i16(x: torch.Tensor, lsb: int, length: int) -> torch.Tensor:
    mask = (1 << length) - 1
    return ((x.to(torch.int32) & 0xFFFF) >> lsb) & mask

def _safe_scale_div(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(scale, min=eps)

def _lookup_with_interp(lut: torch.Tensor, x: float, x_min: float, x_max: float) -> float:
    """Linear interpolation lookup in LUT."""
    LUT_SIZE = lut.shape[0]
    x = max(x_min, min(x_max, x))
    
    # Normalize to [0, 1]
    normalized = (x - x_min) / (x_max - x_min)
    # Map to LUT index
    idx_float = normalized * (LUT_SIZE - 1)
    idx_lo = int(math.floor(idx_float))
    idx_hi = min(idx_lo + 1, LUT_SIZE - 1)
    
    frac = idx_float - idx_lo
    val_lo = float(lut[idx_lo])
    val_hi = float(lut[idx_hi])
    
    return val_lo + frac * (val_hi - val_lo)

# ----- Abstract Base Class for Quantizable Modules -----

class QuantizableModule(nn.Module):
    """
    An abstract base class for all quantizable modules.
    It handles the state management for calibration and fake quantization modes
    to reduce code duplication.
    """
    def __init__(self):
        super(QuantizableModule, self).__init__()
        self.calibrate_mode = False
        self.fakequant_mode = False
        self.use_lut_inference = False

    def stop_calibrate(self):
        self.calibrate_mode = False

    def start_calibrate(self):
        self.calibrate_mode = True

    def disable_fakequant(self):
        self.fakequant_mode = False

    def enable_fakequant(self):
        self.fakequant_mode = True

    def enable_lut_inference(self):
        self.use_lut_inference = True

    def disable_lut_inference(self):
        self.use_lut_inference = False

# ----- Linear -----

class QLinear(QuantizableModule):
    weight_scale: Any
    bias_scale: Any
    input_scale: Any
    output_scale: Any
    input_zero: Any
    output_zero: Any
    fused_scale: Any
    weight_int: Any
    bias_int: Any

    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias: bool = True, 
                 weight_bit: int = 8,
                 bias_bit: int = 8, # 32,
                 act_bit: int = 8,
                 act_quant_mode: str = "sym",
                 act_per_token: bool = False,
                 input_act_per_token=None,
                 output_act_per_token=None,
                 wgt_per_channel: bool = False):
        super(QLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.act_bit = act_bit
        self.act_quant_mode = act_quant_mode

        # Backward-compatible behavior:
        # - If input/output flags are not provided, fall back to legacy act_per_token.
        # - If they are provided, allow all 4 combinations.
        if input_act_per_token is None:
            input_act_per_token = act_per_token
        if output_act_per_token is None:
            output_act_per_token = act_per_token

        self.input_act_per_token = bool(input_act_per_token)
        self.output_act_per_token = bool(output_act_per_token)
        self.wgt_per_channel = wgt_per_channel

        # can't be used at the same time
        assert not(wgt_per_channel and (self.input_act_per_token or self.output_act_per_token)) and "pt and pc can't be used at the same time"

        ## PARAM QUANT DEFINITION ##
        if self.wgt_per_channel:
            wb_quant_param_shape = (out_features, )
        else:
            wb_quant_param_shape = (1, )

        # the weight and bias of nn.Linear in this place store the original weight and bias in the FP model
        # here we still use float formal to store the int value, but then we'll have a method to really convert it to true int formal
        self.register_buffer("weight_int", torch.zeros_like(self.weight)) # currently not register_parameter
        self.register_buffer("weight_scale", torch.zeros(wb_quant_param_shape))
        if bias:
            self.register_buffer("bias_int", torch.zeros_like(self.bias))
            # bias_scale == fused_scale * output_scale == input_scale * weight_scale
            self.register_buffer("bias_scale", torch.zeros(wb_quant_param_shape))
        else:
            self.register_buffer("bias_int", None)
            self.register_buffer("bias_scale", None)

        # input and output can be set sym/asym and pertoken/pertensor
        # NOTICE: we can't assume that output_scale == bias_scale == input_scale * weight_scale, because this will make the output be a int32 number
        input_quant_param_shape = (1,)
        output_quant_param_shape = (1,)
        # fused_scale may become per-token during calibrate(); use placeholder here.
        fused_scale_quant_param_shape = (1,)

        ## ACT QUANT DEFINITION ##
        self.register_buffer("input_scale", torch.zeros(input_quant_param_shape))
        self.register_buffer("output_scale", torch.zeros(output_quant_param_shape))
        # fused_scale = input_scale * weight_scale / output_scale
        # indeed, fused_scale comes from the fusing of QLinear and QAct
        self.register_buffer("fused_scale", torch.zeros(fused_scale_quant_param_shape))
        if act_quant_mode == "sym":
            self.register_buffer("input_zero", None)
            self.register_buffer("output_zero", None)
        elif act_quant_mode == "asym":
            self.register_buffer("input_zero", torch.zeros(input_quant_param_shape))
            self.register_buffer("output_zero", torch.zeros(output_quant_param_shape))
        else:
            raise NotImplementedError("unsupported act quant mode: {}".format(act_quant_mode))

    def calibrate(self, x_float):
        y = self.forward_float(x_float)

        w_scale, _ = max_min_quantize_params(
            input_tensor=self.weight.data,
            bitwidth=self.weight_bit,
            quant_mode="sym", # weight强制sym
            per_channel=self.wgt_per_channel,
            is_weight=True,
        )

        x_scale, x_zero = max_min_quantize_params(
            input_tensor=x_float,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.input_act_per_token,
            is_weight=False,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.output_act_per_token,
            is_weight=False,
        )

        # no bias_zero
        # NOTICE: since perchannel and pertoken would not be used at the same time, here is secure
        # TODO: bias per_token need duplicate and vary in token dimension???
        
        # b_scale = x_scale * w_scale
        b_scale, _ = max_min_quantize_params(
            input_tensor=self.bias.data,
            bitwidth=self.bias_bit,
            quant_mode="sym",
            per_channel=False, # No per-channel
            is_weight=False,
        )

        # insert quant params (use buffer replacement to allow dynamic shapes)
        self.input_scale = x_scale
        self.output_scale = y_scale
        self.fused_scale = x_scale * w_scale / y_scale
        self.weight_scale = w_scale
        self.bias_scale = b_scale
        self.weight_int = symmetric_linear_quantize(
            bits=self.weight_bit, 
            input=self.weight.data, 
            scale=w_scale, 
            is_weight=True,
        )
        if self.bias is not None:
            # NOTICE: No need to add bias if no bias
            # but bias_scale can exist even there is no bias
            # TODO: !!!!!!!!!!!!!!bias_bit must 32???
            self.bias_int = symmetric_linear_quantize(
                bits=self.bias_bit, 
                input=self.bias.data, 
                scale=b_scale, 
                is_weight=True,
            )

        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.input_zero = x_zero
            self.output_zero = y_zero
        
        return y

    def forward_int(self, x_int):
        def _view_1d_param(param: torch.Tensor, ref: torch.Tensor):
            if param is None:
                return None
            if param.numel() == 1:
                return param
            # per-token on dim=1
            if ref.ndim == 3 and param.numel() == ref.shape[1]:
                return param.view(1, -1, 1)
            # per-feature on last dim
            if ref.ndim == 3 and param.numel() == ref.shape[-1]:
                return param.view(1, 1, -1)
            if ref.ndim == 2 and param.numel() == ref.shape[0]:
                return param.view(-1, 1)
            if ref.ndim == 2 and param.numel() == ref.shape[1]:
                return param.view(1, -1)
            return param

        if self.act_quant_mode == "asym" and self.input_zero is not None:
            x_int = x_int - _view_1d_param(self.input_zero, x_int)

        xw_scale = self.input_scale * self.weight_scale
        bias_num = None
        if self.bias_int is not None and self.bias_scale is not None:
            bias_num = (self.bias_int * self.bias_scale).to(torch.float32)

        # Linear without bias first (works for 2D/3D inputs)
        o_int = F.linear(x_int, self.weight_int, None)

        # Add bias_term with correct broadcasting
        if bias_num is not None:
            if xw_scale.numel() == 1:
                o_int = o_int + (bias_num / xw_scale).view(1, *([1] * (o_int.ndim - 2)), -1)
            elif x_int.ndim == 3 and xw_scale.numel() == x_int.shape[1]:
                # per-token input scale
                o_int = o_int + (bias_num.view(1, 1, -1) / xw_scale.view(1, -1, 1))
            elif xw_scale.numel() == self.out_features:
                # per-output-feature scale
                o_int = o_int + (bias_num.view(1, *([1] * (o_int.ndim - 2)), -1) / xw_scale.view(1, *([1] * (o_int.ndim - 2)), -1))
            else:
                o_int = o_int + (bias_num / xw_scale)

        fused_scale_view = _view_1d_param(self.fused_scale, o_int)
        o_int = torch.round(o_int * fused_scale_view)

        if self.act_quant_mode == "asym" and self.output_zero is not None:
            o_int = o_int + _view_1d_param(self.output_zero, o_int)

        return o_int
    
    def forward_int_lut(self, x_int):
        """
        Integer QLinear with fused scale lookup variant.
        For linear layers, LUT is less applicable; this uses pseudo-quantized integer arithmetic.
        Forward path: i8 -> dequantize -> matmul -> quantize -> i8
        """
        def _view_1d_param(param: torch.Tensor, ref: torch.Tensor):
            if param is None:
                return None
            if param.numel() == 1:
                return param
            if ref.ndim == 3 and param.numel() == ref.shape[1]:
                return param.view(1, -1, 1)
            if ref.ndim == 3 and param.numel() == ref.shape[-1]:
                return param.view(1, 1, -1)
            if ref.ndim == 2 and param.numel() == ref.shape[0]:
                return param.view(-1, 1)
            if ref.ndim == 2 and param.numel() == ref.shape[1]:
                return param.view(1, -1)
            return param

        if self.act_quant_mode == "asym" and self.input_zero is not None:
            x_int = x_int - _view_1d_param(self.input_zero, x_int)

        acc = F.linear(x_int, self.weight_int, None)

        drop_bits = 8
        acc_drop = torch.floor(acc / (2 ** drop_bits))
        fused_scale_view = _view_1d_param(self.fused_scale, acc_drop) * (2 ** drop_bits)
        o_int_lut = torch.round(acc_drop * fused_scale_view)
        o_int_lut = torch.clamp(o_int_lut, -2 ** (self.act_bit - 1), 2 ** (self.act_bit - 1) - 1)

        if self.bias_int is not None and self.bias_scale is not None:
            bias_num = (self.bias_int * self.bias_scale).to(torch.float32)
            if self.output_scale.numel() == 1:
                bias_q = torch.round(bias_num / self.output_scale).view(1, *([1] * (o_int_lut.ndim - 2)), -1)
            elif o_int_lut.ndim == 3 and self.output_scale.numel() == o_int_lut.shape[1]:
                bias_q = torch.round(bias_num.view(1, 1, -1) / self.output_scale.view(1, -1, 1))
            elif self.output_scale.numel() == self.out_features:
                bias_q = torch.round(
                    bias_num.view(1, *([1] * (o_int_lut.ndim - 2)), -1)
                    / self.output_scale.view(1, *([1] * (o_int_lut.ndim - 2)), -1)
                )
            else:
                bias_q = torch.round(bias_num / self.output_scale)
            o_int_lut = o_int_lut + bias_q
            o_int_lut = torch.clamp(o_int_lut, -2 ** (self.act_bit - 1), 2 ** (self.act_bit - 1) - 1)

        if self.act_quant_mode == "asym" and self.output_zero is not None:
            o_int_lut = o_int_lut + _view_1d_param(self.output_zero, o_int_lut)

        return o_int_lut
    
    def forward_float(self, x_float):
        # w_float = self.weight_int * self.weight_scale.view(-1, 1)
        # b_float = self.bias_int * self.bias_scale
        # return F.linear(x_float, w_float, b_float)

        # NOTICE: use this to avoid when the weight_int not ready
        return F.linear(x_float, self.weight, self.bias)

    def forward(self, x_float):
        # i think calibrate is more significant
        if self.calibrate_mode:
            return self.calibrate(x_float)
        # now fake quant
        elif self.fakequant_mode:
            def _view_1d_param(param: torch.Tensor, ref: torch.Tensor):
                if param is None:
                    return None
                if param.numel() == 1:
                    return param
                if ref.ndim == 3 and param.numel() == ref.shape[1]:
                    return param.view(1, -1, 1)
                if ref.ndim == 2 and param.numel() == ref.shape[0]:
                    return param.view(-1, 1)
                if ref.ndim == 2 and param.numel() == ref.shape[1]:
                    return param.view(1, -1)
                return param

            in_scale = _view_1d_param(self.input_scale, x_float)
            x_int = torch.clamp(
                torch.round(x_float / in_scale),
                -2 ** (self.act_bit - 1),
                2 ** (self.act_bit - 1) - 1,
            )
            y_int = self.forward_int_lut(x_int=x_int) if self.use_lut_inference else self.forward_int(x_int=x_int)
            out_scale = _view_1d_param(self.output_scale, y_int)
            return y_int * out_scale
        else:
            return self.forward_float(x_float)

    
    def copy_from(
        self,
        linear: nn.Linear,
    ):
        self.weight.data = linear.weight.data
        if linear.bias is not None and self.bias is None:
            raise ValueError("This QLinear instance has no bias setting")
        elif linear.bias is None and self.bias is not None:
            self.bias.data = torch.zeros_like(self.bias.data)
        else:
            self.bias.data = linear.bias
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, linear: nn.Linear, 
                 weight_bit=8,
                 bias_bit=32,
                 act_bit=8,
                 act_quant_mode="sym",
                 act_per_token=False,
                 input_act_per_token=None,
                 output_act_per_token=None,
                 wgt_per_channel=False):
        return cls(linear.in_features, 
                   linear.out_features,
                   weight_bit=weight_bit,
                   bias_bit=bias_bit,
                   act_bit=act_bit,
                   act_quant_mode=act_quant_mode,
                   act_per_token=act_per_token,
                   input_act_per_token=input_act_per_token,
                   output_act_per_token=output_act_per_token,
                   wgt_per_channel=wgt_per_channel,
                   ).copy_from(linear)

# ----- QConv2d -----

# NOTE: Currently not used.
# TODO: bias update same as QLinear
class QConv2d(QuantizableModule):
    weight_scale: Any
    bias_scale: Any
    input_scale: Any
    output_scale: Any
    input_zero: Any
    output_zero: Any
    fused_scale: Any
    weight_int: Any
    bias_int: Any

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = True, 
                 weight_bit: int = 8,
                 bias_bit: int = 32,
                 act_bit: int = 8,
                 act_quant_mode: str = "sym",
                 wgt_per_channel: bool = False):
        super(QConv2d, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.act_bit = act_bit
        self.act_quant_mode = act_quant_mode
        self.wgt_per_channel = wgt_per_channel

        if self.wgt_per_channel:
            wb_quant_param_shape = (out_channels, )
        else:
            wb_quant_param_shape = (1, )

        # weight and bias is always sym-quant
        self.register_buffer("weight_int", torch.zeros_like(self.weight))
        self.register_buffer("weight_scale", torch.zeros(wb_quant_param_shape))
        if bias:
            self.register_buffer("bias_int", torch.zeros_like(self.bias))
            # bias_scale == input_scale * weight_scale
            self.register_buffer("bias_scale", torch.zeros(wb_quant_param_shape))
        else:
            self.register_buffer("bias", None)
            self.register_buffer("bias_scale", None)

        fused_quant_param_shape = (1 * self.bias_scale.numel(), )

        self.register_buffer("input_scale", torch.zeros(1, ))
        self.register_buffer("output_scale", torch.zeros(1, ))
        # fused_scale = input_scale * weight_scale / output_scale
        self.register_buffer("fused_scale", torch.zeros(fused_quant_param_shape))
        if act_quant_mode == "sym":
            self.register_buffer("input_zero", None)
            self.register_buffer("output_zero", None)
        elif act_quant_mode == "asym":
            self.register_buffer("input_zero", torch.zeros((1, )))
            self.register_buffer("output_zero", torch.zeros((1, )))
        else:
            raise NotImplementedError("unsupported act quant mode: {}".format(act_quant_mode))

    def calibrate(self, x_float):
        y = self.forward_float(x_float)

        w_scale, _ = max_min_quantize_params(
            input_tensor=self.weight.data,
            bitwidth=self.weight_bit,
            quant_mode="sym", # weight强制sym
            per_channel=self.wgt_per_channel,
            is_weight=True,
        )

        x_scale, x_zero = max_min_quantize_params(
            input_tensor=x_float,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=False,
            is_weight=False,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=False,
            is_weight=False,
        )

        b_scale = x_scale * w_scale
        # no bias_zero

        # insert quant params
        self.input_scale.data = x_scale
        self.output_scale.data = y_scale
        self.weight_scale.data = w_scale
        self.bias_scale.data = b_scale
        self.fused_scale.data = b_scale / y_scale
        self.weight_int.data = symmetric_linear_quantize(
            bits=self.weight_bit, 
            input=self.weight.data, 
            scale=w_scale, 
            is_weight=True,
        )
        self.bias_int.data = symmetric_linear_quantize(
            bits=self.bias_bit, 
            input=self.bias.data, 
            scale=b_scale, 
            is_weight=True,
        )
        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.input_zero.data = x_zero
            self.output_zero.data = y_zero
        
        return y
    
    def forward_float(self, x_float):
        # w_float = self.weight_int * self.weight_scale.view(-1, 1, 1, 1)
        # b_float = self.bias_int * self.bias_scale
        # return F.conv2d(
        #     x_float, 
        #     w_float, 
        #     b_float,
        #     self.stride,
        #     self.padding,
        #     self.dilation,
        #     self.groups,
        # )

        # NOTICE: use this to avoid the weight_int not ready
        return F.conv2d(
                x_float, 
                self.weight, 
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
    
    def forward_int(self, x_int):
        if self.act_quant_mode == "asym":
            # asymmetric
            x_int = x_int - self.input_zero

        o_int = F.conv2d(
            x_int, 
            self.weight_int, 
            self.bias_int,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        o_int = torch.round(o_int * self.fused_scale.reshape(1, -1, 1, 1))
        
        if self.act_quant_mode == "asym":
            o_int = o_int + self.output_zero

        return o_int
    
    def forward_int_lut(self, x_int):
        """
        Integer QConv2d with pseudo-quantized arithmetic.
        Forward path: i8 -> dequantize -> conv -> quantize -> i8
        """
        return self.forward_int(x_int)
    
    def forward(self, x_float):
        if self.calibrate_mode:
            return self.calibrate(x_float)
        elif self.fakequant_mode:
            x_int = torch.clamp(torch.round(x_float / self.input_scale), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)
            y_int = self.forward_int_lut(x_int=x_int) if self.use_lut_inference else self.forward_int(x_int=x_int)
            return y_int * self.output_scale[:, None, None]
        else:
            return self.forward_float(x_float)

    def copy_from(
        self,
        conv2d: nn.Conv2d,
    ):
        self.weight.data = conv2d.weight.data
        if conv2d.bias is not None and self.bias is None:
            raise ValueError("This Conv2d instance has no bias setting")
        elif conv2d.bias is None and self.bias is not None:
            self.bias.data = torch.zeros_like(self.bias.data)
        else:
            self.bias.data = conv2d.bias
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, 
                      conv2d: nn.Conv2d,
                      weight_bit=8,
                    bias_bit=32,
                    act_bit=8,
                    act_quant_mode="sym",
                    wgt_per_channel=False
                    ):
        return cls(
            conv2d.in_channels, 
            conv2d.out_channels,
            conv2d.kernel_size,
            conv2d.stride,
            conv2d.padding,
            conv2d.dilation,
            conv2d.groups,
            conv2d.bias is not None,
            weight_bit=weight_bit,
            bias_bit=bias_bit,
            act_bit=act_bit,
            act_quant_mode=act_quant_mode,
            wgt_per_channel=wgt_per_channel,
        ).copy_from(conv2d)

class QFFN(QuantizableModule):
    def __init__(self, 
                 n_embd, 
                 hidden_dim, 
                 output_dim
                 ):
        super(QFFN, self).__init__()
        self.fc1 = QLinear(n_embd, hidden_dim) # TODO: not just default quant parameter
        self.fc2 = QLinear(hidden_dim, output_dim)
        self.activation = IntGELU()

    def start_calibrate(self):
        self.fc1.start_calibrate()
        self.fc2.start_calibrate()

    def stop_calibrate(self):
        self.fc1.stop_calibrate()
        self.fc2.stop_calibrate()

    def disable_fakequant(self):
        self.fc1.disable_fakequant()
        self.fc2.disable_fakequant()

    def enable_fakequant(self):
        self.fc1.enable_fakequant()
        self.fc2.enable_fakequant()

    def enable_lut_inference(self):
        self.fc1.enable_lut_inference()
        self.activation.enable_lut_inference()
        self.fc2.enable_lut_inference()

    def disable_lut_inference(self):
        self.fc1.disable_lut_inference()
        self.activation.disable_lut_inference()
        self.fc2.disable_lut_inference()

    def forward_int(self, x_int):
        x_int = self.fc1.forward_int(x_int)
        x_int = self.activation.forward_int(x_int)
        x_int = self.fc2.forward_int(x_int)
        return x_int
    
    def forward_int_lut(self, x_int):
        """Integer FFN with LUT-based activation."""
        x_int = self.fc1.forward_int_lut(x_int)
        x_int = self.activation.forward_int_lut(x_int)
        x_int = self.fc2.forward_int_lut(x_int)
        return x_int

    def forward(self, x):
        # TODO: if you use QFFN, need similarly if-else like other fwd 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    # TODO：should FFN be replaced？

# ----- IntGELU -----

class IntGELU(QuantizableModule):
    input_scale: Any
    output_scale: Any
    input_zero: Any
    output_zero: Any
    gelu_scale: Any
    fused_scale: Any

    def __init__(self, 
            act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,
        ):
        super(IntGELU, self).__init__()
        self.c = 31 # sigmoid capacity, it's node side effect outside the forward of intgelu
        self.n = self.c - act_bit # the capacity of exp, it's no side effect outside the _int_exp function
        self.act_bit = act_bit
        self.act_quant_mode = act_quant_mode
        self.act_per_token = act_per_token

        # input and output can be set sym/asym and per_token/pertensor
        if act_per_token:
            # raise NotImplementedError("unsupported per channel for act currently")
            input_quant_param_shape = (1,)
            output_quant_param_shape = (1,)
            fused_scale_quant_param_shape = (1,)
        else: # per tensor
        # when the setting is per tensor, we can assume that output_scale == bias_scale == input_scale * weight_scale
            input_quant_param_shape = (1,)
            output_quant_param_shape = (1,)
            fused_scale_quant_param_shape = (1,)

        self.register_buffer("input_scale", torch.zeros(input_quant_param_shape)) # when it's per_tensor, its shape can be simply decided; otherwise it's just a placeholder, whose shape can't be instantly decided 
        self.register_buffer("output_scale", torch.zeros(output_quant_param_shape))
        self.register_buffer("gelu_scale", torch.zeros(fused_scale_quant_param_shape))
        # fused_scale = gelu_scale / output_scale
        self.register_buffer("fused_scale", torch.zeros(fused_scale_quant_param_shape))

        if act_quant_mode == "sym":
            self.register_buffer("input_zero", None)
            self.register_buffer("output_zero", None)
        elif act_quant_mode == "asym":
            self.register_buffer("input_zero", torch.zeros(input_quant_param_shape))
            self.register_buffer("output_zero", torch.zeros(output_quant_param_shape))
        else:
            raise NotImplementedError("unsupported act quant mode: {}".format(act_quant_mode))

    def _int_exp(self, x_int, scale):
        x_int = x_int + torch.floor(x_int / 2) - torch.floor(x_int / 2**4)
        x_unit = torch.floor(-1.0 / scale)
        x_int = torch.max(x_int, self.n * x_unit)

        q = torch.floor(x_int / x_unit)
        r = x_int - q * x_unit
        exp_int = r/2 - x_unit
        exp_int = torch.clamp(torch.floor(exp_int * 2 ** (self.n - q)), min=0)

        # the scale of exp_int is scale / 2 ** (self.n)

        return exp_int
    
    def __int_exp(self, int_angle, pre_scale, m=16, iterations=10):
        int_angle = int_angle << 4
        # 预计算的反正切值表 atanh(2^(-i))，单位为弧度
        artanh_table = [math.inf] + [
            torch.round(math.atanh(2**(-i)) * 16 / pre_scale) for i in range(1, iterations + 1)
        ]
        scale_factor = 1.205136358446461
        # 初始值
        x_current = torch.ones_like(int_angle, dtype=torch.int) * round((1 << m) * scale_factor)
        y_current = torch.zeros_like(int_angle, dtype=torch.int)
        z_current = int_angle
        # CORDIC迭代
        k = 4
        for i in range(1, iterations + 1):
            # 确定旋转方向
            d = (z_current > 0).int() * 2 - 1
            # 旋转操作（不使用乘法，通过移位实现2^(-i)）
            x_next = x_current + d * (y_current >> i) # (y_current >> i)
            y_next = y_current + d * (x_current >> i) # (x_current >> i)
            z_next = z_current - d * artanh_table[i]
            if i == k:
                x_next = x_next + d * (y_next >> i)
                y_next = y_next + d * (x_next >> i)
                z_next = z_next - d * artanh_table[i]
                k = 3 * k + 1
            # 更新当前值
            x_current, y_current, z_current = x_next, y_next, z_next
        # 返回最终角度，加上初始调整
        return x_current + y_current

    def calibrate(self, x_float):
        y = self.forward_float(x_float)

        x_scale, x_zero = max_min_quantize_params(
            input_tensor=x_float,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token,
            is_weight=False,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token,
            is_weight=False,
        )

        self.input_scale.data = x_scale
        self.output_scale.data = y_scale
        self.gelu_scale.data = x_scale / 2 ** (self.act_bit - 1)
        self.fused_scale.data = self.gelu_scale / self.output_scale

        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.input_zero.data = x_zero
            self.output_zero.data = y_zero

        return y
    
    def forward_float(self, x_float):
        return F.gelu(x_float)

    def forward_int(self, x_int):
        if self.act_quant_mode == "asym":
            x_int = x_int - self.input_zero

        pre_x_int = x_int
        x_int_max, _ = torch.max(x_int, dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_scale = self.input_scale * 1.702
        # exp_int = self.__int_exp(x_int.int(), exp_scale, self.c).float()
        # exp_int_max = self.__int_exp(-x_int_max.int(), exp_scale, self.c).float()
        exp_int = self._int_exp(x_int, exp_scale[None, :, None]) # [B, S, H]
        exp_int_max = self._int_exp(-x_int_max, exp_scale[None, :, None]) # [B, S, H]
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**self.c-1)

        factor = torch.floor((2**self.c-1) / exp_int_sum)
        # this part makes the scale of exp no need
        sigmoid_int = torch.floor(exp_int * factor / 2 ** (self.c-self.act_bit+1))
        # and the self.act_bit is directly ensure the output_bit so that the extra scale conversion is not needed
        y_int = pre_x_int * sigmoid_int

        # gelu_scale == input_scale / 2 ** (act_bit - 1) when it's per_tensor for act
        # 1 / 2 ** (act_bit - 1) is the scale of the sigmoid

        y_int = torch.round(y_int * self.fused_scale[None, :, None])

        if self.act_quant_mode == "asym":
            y_int = y_int + self.output_zero

        return y_int
    
    def forward_int_lut(self, x_int):
        """
        Integer GELU using LUT (Lookup Table).
        Forward path: i8 -> dequantize -> fixed-point -> lookup -> quantize -> i8
        """
        _init_vector_luts()
        
        if self.act_quant_mode == "asym":
            x_int = x_int - self.input_zero

        x_float = x_int.float() * self.input_scale[None, :, None]

        # LUT only applies to [-4, 4]; outside this range use piecewise behavior:
        # x < -4 -> 0, x > 4 -> x
        x_fix_q13 = torch.round(torch.clamp(x_float, -4.0, 4.0) * (2 ** 13)).to(torch.int32)
        x_fix_q13 = _clip_i16(x_fix_q13)

        idx = _extract_bits_i16(x_fix_q13, 6, 10).long()
        gelu_fix_q13 = _g_gelu_rev_lut[idx].to(torch.int32)

        gelu_lut_float = gelu_fix_q13.float() / (2 ** 13)
        gelu_float = gelu_lut_float
        # optional: not saturate
        # gelu_float = torch.where(
        #     x_float < -4.0,
        #     torch.zeros_like(x_float),
        #     torch.where(x_float > 4.0, x_float, gelu_lut_float),
        # )

        y_int_lut = torch.round(_safe_scale_div(gelu_float, self.output_scale[None, :, None]))
        y_int_lut = torch.clamp(y_int_lut, -2**(self.act_bit-1), 2**(self.act_bit-1)-1)
        
        if self.act_quant_mode == "asym":
            y_int_lut = y_int_lut + self.output_zero

        return y_int_lut

    def forward(self, x_float):
        if self.calibrate_mode:
            return self.calibrate(x_float)
        elif self.fakequant_mode:
            x_int = torch.clamp(torch.round(x_float / self.input_scale[None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)
            y_int = self.forward_int_lut(x_int=x_int) if self.use_lut_inference else self.forward_int(x_int=x_int)
            return y_int * self.output_scale[None, :, None]
        else:
            return self.forward_float(x_float)
    
    def copy_from(self, gelu):
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, gelu: nn.GELU, act_bit=8,
            act_quant_mode="sym",
            act_per_token=False):
        return cls(act_bit=act_bit, act_quant_mode=act_quant_mode, act_per_token=act_per_token).copy_from(gelu)

# ----- IntSoftmax -----

class IntSoftmax(QuantizableModule):
    input_scale: Any
    output_scale: Any
    input_zero: Any
    output_zero: Any
    softmax_scale: Any
    fused_scale: Any

    def __init__(self, 
            dim: int = -1,
            in_act_bit: int = 8,
            softmax_act_bit: int = 16,
            out_act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,
            act_per_head: bool = False,
        ):
        super(IntSoftmax, self).__init__()
        self.debug_count = 2
        self.dim = dim
        self.c = 32 # sigmoid capacity, it's node side effect outside the forward of intgelu
        self.n = self.c - softmax_act_bit # the capacity of exp, it's no side effect outside the _int_exp function
        self.in_act_bit = in_act_bit
        self.softmax_act_bit = softmax_act_bit
        self.out_act_bit = out_act_bit
        self.act_quant_mode = act_quant_mode
        self.act_per_token = act_per_token
        self.act_per_head = act_per_head

        # TODO: WIP: If Softmax in MHA contains multiple instances of quant param， but it's not perferred now
        self.multiple_heads = False
        self.head_index = 0

        # input and output can be set sym/asym and perchannel/pertensor
        if act_per_token:
            input_quant_param_shape = (1,)
            output_quant_param_shape = (1,)
            # NOTICE: assume per-token and per-head not be used meanwhile
            act_per_head = False
        elif act_per_head:
            # TODO: 
            self.act_per_head = act_per_head
            raise NotImplementedError("Not Implemented per-head")
        else: # per tensor
        # when the setting is per tensor, we can assume that output_scale == bias_scale == input_scale * weight_scale
            input_quant_param_shape = (1,)
            output_quant_param_shape = (1,)

        self.register_buffer("input_scale", torch.zeros(input_quant_param_shape)) # when it's per_tensor, its shape can be simply decided; otherwise it's just a placeholder, whose shape can't be instantly decided 
        self.register_buffer("softmax_scale", torch.zeros(output_quant_param_shape))
        self.register_buffer("output_scale", torch.zeros(output_quant_param_shape))

        if act_quant_mode == "sym":
            self.register_buffer("input_zero", None)
            self.register_buffer("output_zero", None)
        elif act_quant_mode == "asym":
            self.register_buffer("input_zero", torch.zeros(input_quant_param_shape))
            self.register_buffer("output_zero", torch.zeros(output_quant_param_shape))
        else:
            raise NotImplementedError("unsupported act quant mode: {}".format(act_quant_mode))
        
        if out_act_bit != softmax_act_bit:
            self.register_buffer("fused_scale", torch.zeros(output_quant_param_shape))
        #     self.qact = QAct(
        #         in_act_bit=softmax_act_bit,
        #         out_act_bit=out_act_bit,
        #         in_act_quant_mode="sym", # The result of Softmax is absolutely sym
        #         out_act_quant_mode=act_quant_mode,
        #     )
        #     # TODO: add the assignment of qact's params under different act_quant_mode
        #     # self.qact.input_scale.data = self.softmax_scale.data.detach()

    def _int_exp(self, x_int, scale):
        x_int = x_int + torch.floor(x_int / 2) - torch.floor(x_int / 2**4)
        x_unit = torch.floor(-1.0 / scale)
        x_int = torch.max(x_int, self.n * x_unit)

        q = torch.floor(x_int / x_unit)
        r = x_int - q * x_unit
        exp_int = r/2 - x_unit
        exp_int = torch.clamp(torch.floor(exp_int * 2 ** (self.n - q)), min=0)

        # the scale of exp_int is scale / 2 ** (self.n)
        # the bit of exp_int is 2 ** self.n

        return exp_int

    def calibrate(self, x_float):
        y = self.forward_float(x_float)

        x_scale, x_zero = max_min_quantize_params(
            input_tensor=x_float,
            bitwidth=self.in_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token or self.act_per_head,
            is_weight=False,
            is_seq_x=self.act_per_token, # per-channel must be True to use this, act_per_token means it reduces on dim=2, otherwise is act_per_head which reduces on dim = 1
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.out_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token or self.act_per_head,
            is_weight=False,
            is_seq_x=self.act_per_token,
        )

        self.input_scale.data = x_scale
        self.output_scale.data = y_scale
        self.softmax_scale.data = torch.Tensor([1 / 2 ** (self.softmax_act_bit - 1)])
        if self.softmax_act_bit != self.out_act_bit:
            # self.qact.fused_scale.data = self.softmax_scale / self.output_scale
            self.fused_scale.data = self.softmax_scale / self.output_scale
        else: # TODO: ??maybe useful, please according to directly int-to-int forward formula
            self.output_scale.data = self.softmax_scale

        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.input_zero.data = x_zero
            self.output_zero.data = y_zero
            # self.qact.output_zero.data = y_zero

        return y

    def forward_int(self, x_int):
        if self.act_quant_mode == "asym":
            x_int = x_int - self.input_zero

        x_int_max, _ = torch.max(x_int, dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_scale = self.input_scale[None, None, :, None]
        exp_int = self._int_exp(x_int, exp_scale)
        exp_int_sum = torch.sum(exp_int, dim=self.dim, keepdim=True)

        exp_int_sum.clamp_max_(2**self.c-1)

        factor = torch.floor((2**self.c-1) / exp_int_sum)
        # this part makes the scale of exp no need
        softmax_int = torch.floor(exp_int * factor / 2 ** (self.c-self.softmax_act_bit+1))

        # softmax_scale == 1 / 2 ** (softmax_act_bit-1) when it's per_tensor for act
        # 1 / 2 ** (softmax_act_bit-1) comes from the scale of the calculation of softmax
        # the bit of softmax is `softmax_act_bit`

        if self.out_act_bit != self.softmax_act_bit:
            # NOTICE: Not Use QAct 
            # softmax_int = self.qact.forward_int(softmax_int)
            softmax_int = softmax_int * self.fused_scale[None, None, :, None]

        # finally, add zero back if asymmetric
        if self.act_quant_mode == "asym":
            softmax_int = softmax_int + self.output_zero
        
        return softmax_int

    def forward_int_lut(self, x_int):
        """
        Integer Softmax using LUT.
        Forward path: i8 -> dequantize -> exp_lut -> sum -> softmax -> quantize -> i8
        """
        _init_vector_luts()
        
        if self.act_quant_mode == "asym":
            x_int = x_int - self.input_zero

        x_int = x_int - torch.max(x_int, dim=self.dim, keepdim=True)[0] # 127 # torch.max(x_int, dim=self.dim, keepdim=True)[0]

        # i8 -> dequant(float) -> fixed(3,12)
        x_float = x_int.float() * self.input_scale[None, None, :, None]
        x_fix_q12 = torch.round(x_float * (2 ** 12)).to(torch.int32)
        x_fix_q12 = _clip_i16(x_fix_q12)

        # driver_sim stage1: idx1=bits[14:5], ex_sum
        idx1 = _extract_bits_i16(x_fix_q12, 5, 10).long()
        ex_rev_lut_i32 = _g_softmax_ex_rev_lut.to(torch.int32)
        ex_val = ex_rev_lut_i32[idx1]  # Q0.16
        ex_sum = torch.sum(ex_val, dim=self.dim, keepdim=True)

        # driver_sim / softmax_quant.py: loge_idx = ex_sum >> (16 + 8 - 10)
        loge_idx = torch.clamp((ex_sum >> 14).to(torch.long), 0, len(_g_softmax_loge_lut) - 1)
        loge_val = _g_softmax_loge_lut[loge_idx].to(torch.int32)  # Q3.12

        # driver_sim stage2: idx2 from (x - loge)
        minus_loge = _clip_i16(x_fix_q12 - loge_val)
        idx2 = _extract_bits_i16(minus_loge, 5, 10).long()
        out_fix_q16 = ex_rev_lut_i32[idx2]  # Q0.16

        # fixed -> float -> i8
        out_float = out_fix_q16.float() / (2 ** 16)
        softmax_int_lut = torch.round(_safe_scale_div(out_float, self.output_scale[None, None, :, None]))
        softmax_int_lut = torch.clamp(softmax_int_lut, -2**(self.out_act_bit-1), 2**(self.out_act_bit-1)-1)
        
        if self.act_quant_mode == "asym":
            softmax_int_lut = softmax_int_lut + self.output_zero
        
        return softmax_int_lut

    def forward(self, x_float):
        if self.calibrate_mode:
            return self.calibrate(x_float)
        elif self.fakequant_mode:
            self.debug_count += 1
            x_int = torch.clamp(torch.round(x_float / self.input_scale[None, None, :, None]), -2**(self.in_act_bit-1), 2**(self.in_act_bit-1)-1)
            y_int = self.forward_int_lut(x_int=x_int) if self.use_lut_inference else self.forward_int(x_int=x_int)
            tmp = y_int * self.output_scale[None, None, :, None]
            if self.debug_count == 1:
                print("Debug Info of IntSoftmax:")
                print("Input Scale:", self.input_scale)
                print("Softmax Scale:", self.softmax_scale)
                print("Output Scale:", self.output_scale)
                if self.out_act_bit != self.softmax_act_bit:
                    print("Fused Scale:", self.fused_scale)
                print("Input Sample:", x_float)
                print("Output Sample:", tmp)
            return tmp
        else:
            return self.forward_float(x_float)

    def forward_float(self, x_float):
        return F.softmax(x_float, dim=-1)
    
    def copy_from(self, softmax):
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, softmax: nn.Softmax,
            dim: int = -1,
            in_act_bit: int = 8,
            softmax_act_bit: int = 16,
            out_act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,
            act_per_head: bool = False,
        ):
        return cls(
            dim=dim,
            in_act_bit=in_act_bit,
            softmax_act_bit=softmax_act_bit,
            out_act_bit=out_act_bit,
            act_quant_mode=act_quant_mode,
            act_per_token=act_per_token,
            act_per_head=act_per_head,
        ).copy_from(softmax)


class IntSoftmaxWithMask(IntSoftmax):
    def _view_param(self, param: torch.Tensor, x: torch.Tensor):
        if param is None:
            return None
        if param.numel() == 1:
            return param.view(*([1] * x.ndim))
        if x.ndim == 4 and param.numel() == x.shape[2]:
            return param[None, None, :, None]
        if x.ndim == 3 and param.numel() == x.shape[1]:
            return param[None, :, None]
        return param.reshape(*([1] * (x.ndim - 1)), -1)

    def _get_contiguous_boundary_len(self, attention_mask: torch.Tensor, seq_len: int):
        if attention_mask is None:
            return None
        if attention_mask.shape[-1] != seq_len:
            return None

        flat = attention_mask.detach().reshape(-1, seq_len)
        if flat.shape[0] == 0:
            return None

        row0 = flat[0]
        valid_values = (row0 == 0) | torch.isneginf(row0)
        if not bool(valid_values.all()):
            return None

        if flat.shape[0] > 1:
            all_valid = ((flat == 0) | torch.isneginf(flat)).all()
            if not bool(all_valid):
                return None
            if not bool((flat == row0.unsqueeze(0)).all()):
                return None

        neginf_mask = torch.isneginf(row0)
        if not bool(neginf_mask.any()):
            return seq_len

        first_neginf = int(torch.argmax(neginf_mask.to(torch.int32)).item())
        if first_neginf > 0 and bool(neginf_mask[:first_neginf].any()):
            return None
        if first_neginf < seq_len and bool((~neginf_mask[first_neginf:]).any()):
            return None
        return first_neginf

    def _slice_params_by_boundary(self, valid_len: int, orig_len: int):
        input_scale = self.input_scale
        output_scale = self.output_scale
        input_zero = self.input_zero
        output_zero = self.output_zero

        if input_scale is not None and input_scale.numel() == orig_len:
            input_scale = input_scale[:valid_len]
        if output_scale is not None and output_scale.numel() == orig_len:
            output_scale = output_scale[:valid_len]
        if input_zero is not None and input_zero.numel() == orig_len:
            input_zero = input_zero[:valid_len]
        if output_zero is not None and output_zero.numel() == orig_len:
            output_zero = output_zero[:valid_len]
        return input_scale, output_scale, input_zero, output_zero

    def _forward_int_impl(self, x_int: torch.Tensor, input_scale, output_scale, input_zero, output_zero):
        if self.act_quant_mode == "asym" and input_zero is not None:
            x_int = x_int - self._view_param(input_zero, x_int)

        x_int_max, _ = torch.max(x_int, dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_scale = self._view_param(input_scale, x_int)
        exp_int = self._int_exp(x_int, exp_scale)
        exp_int_sum = torch.sum(exp_int, dim=self.dim, keepdim=True)

        exp_int_sum.clamp_max_(2**self.c - 1)

        factor = torch.floor((2**self.c - 1) / exp_int_sum)
        softmax_int = torch.floor(exp_int * factor / 2 ** (self.c - self.softmax_act_bit + 1))

        if self.out_act_bit != self.softmax_act_bit:
            fused_scale = self._view_param(self.fused_scale, softmax_int)
            softmax_int = softmax_int * fused_scale

        if self.act_quant_mode == "asym" and output_zero is not None:
            softmax_int = softmax_int + self._view_param(output_zero, softmax_int)

        return softmax_int

    def _forward_int_lut_impl(self, x_int: torch.Tensor, input_scale, output_scale, input_zero, output_zero):
        _init_vector_luts()

        if self.act_quant_mode == "asym" and input_zero is not None:
            x_int = x_int - self._view_param(input_zero, x_int)

        x_int = x_int - torch.max(x_int, dim=self.dim, keepdim=True)[0]

        input_scale_view = self._view_param(input_scale, x_int)
        x_float = x_int.float() * input_scale_view
        x_fix_q12 = torch.round(x_float * (2 ** 12)).to(torch.int32)
        x_fix_q12 = _clip_i16(x_fix_q12)

        idx1 = _extract_bits_i16(x_fix_q12, 5, 10).long()
        ex_rev_lut_i32 = _g_softmax_ex_rev_lut.to(torch.int32)
        ex_val = ex_rev_lut_i32[idx1]
        ex_sum = torch.sum(ex_val, dim=self.dim, keepdim=True)

        loge_idx = torch.clamp((ex_sum >> 14).to(torch.long), 0, len(_g_softmax_loge_lut) - 1)
        loge_val = _g_softmax_loge_lut[loge_idx].to(torch.int32)

        minus_loge = _clip_i16(x_fix_q12 - loge_val)
        idx2 = _extract_bits_i16(minus_loge, 5, 10).long()
        out_fix_q16 = ex_rev_lut_i32[idx2]

        out_float = out_fix_q16.float() / (2 ** 16)
        output_scale_view = self._view_param(output_scale, out_float)
        softmax_int_lut = torch.round(_safe_scale_div(out_float, output_scale_view))
        softmax_int_lut = torch.clamp(softmax_int_lut, -2 ** (self.out_act_bit - 1), 2 ** (self.out_act_bit - 1) - 1)

        if self.act_quant_mode == "asym" and output_zero is not None:
            softmax_int_lut = softmax_int_lut + self._view_param(output_zero, softmax_int_lut)

        return softmax_int_lut

    def calibrate(self, x_float, attention_mask=None):
        y = self.forward_float(x_float, attention_mask)

        if attention_mask is not None:
            neginf_mask = torch.isneginf(attention_mask)
            x_stat = torch.where(neginf_mask, torch.zeros_like(x_float), x_float)
        else:
            x_stat = x_float

        x_scale, x_zero = max_min_quantize_params(
            input_tensor=x_stat,
            bitwidth=self.in_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token or self.act_per_head,
            is_weight=False,
            is_seq_x=self.act_per_token,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.out_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token or self.act_per_head,
            is_weight=False,
            is_seq_x=self.act_per_token,
        )

        self.input_scale.data = x_scale
        self.output_scale.data = y_scale
        self.softmax_scale.data = torch.Tensor([1 / 2 ** (self.softmax_act_bit - 1)])
        if self.softmax_act_bit != self.out_act_bit:
            self.fused_scale.data = self.softmax_scale / self.output_scale
        else:
            self.output_scale.data = self.softmax_scale

        if self.act_quant_mode == "asym":
            self.input_zero.data = x_zero
            self.output_zero.data = y_zero

        return y

    def forward_int(self, x_int, attention_mask=None):
        seq_len = x_int.shape[-1]
        valid_len = self._get_contiguous_boundary_len(attention_mask, seq_len)

        input_scale = self.input_scale
        output_scale = self.output_scale
        input_zero = self.input_zero
        output_zero = self.output_zero

        if x_int.ndim == 4 and valid_len is not None and 0 < valid_len < seq_len:
            x_crop = x_int[:, :, :valid_len, :valid_len]
            input_scale, output_scale, input_zero, output_zero = self._slice_params_by_boundary(valid_len, seq_len)
            y_crop = self._forward_int_impl(x_crop, input_scale, output_scale, input_zero, output_zero)
            y_full = torch.zeros_like(x_int)
            y_full[:, :, :valid_len, :valid_len] = y_crop
            return y_full

        return self._forward_int_impl(x_int, input_scale, output_scale, input_zero, output_zero)

    def forward_int_lut(self, x_int, attention_mask=None):
        seq_len = x_int.shape[-1]
        valid_len = self._get_contiguous_boundary_len(attention_mask, seq_len)

        input_scale = self.input_scale
        output_scale = self.output_scale
        input_zero = self.input_zero
        output_zero = self.output_zero

        if x_int.ndim == 4 and valid_len is not None and 0 < valid_len < seq_len:
            x_crop = x_int[:, :, :valid_len, :valid_len]
            input_scale, output_scale, input_zero, output_zero = self._slice_params_by_boundary(valid_len, seq_len)
            y_crop = self._forward_int_lut_impl(x_crop, input_scale, output_scale, input_zero, output_zero)
            y_full = torch.zeros_like(x_int)
            y_full[:, :, :valid_len, :valid_len] = y_crop
            return y_full

        return self._forward_int_lut_impl(x_int, input_scale, output_scale, input_zero, output_zero)

    def forward_float(self, x_float, attention_mask=None):
        if attention_mask is not None:
            x_float = x_float + attention_mask
        return F.softmax(x_float, dim=-1)

    def forward(self, x_float, attention_mask=None):
        if self.calibrate_mode:
            return self.calibrate(x_float, attention_mask)
        elif self.fakequant_mode:
            x_scale_view = self._view_param(self.input_scale, x_float)
            x_int = torch.clamp(
                torch.round(x_float / x_scale_view),
                -2 ** (self.in_act_bit - 1),
                2 ** (self.in_act_bit - 1) - 1,
            )
            if self.use_lut_inference:
                y_int = self.forward_int_lut(x_int=x_int, attention_mask=attention_mask)
            else:
                y_int = self.forward_int(x_int=x_int, attention_mask=attention_mask)
            return y_int * self._view_param(self.output_scale, y_int)
        else:
            return self.forward_float(x_float, attention_mask)

    def copy_from(self, softmax):
        return self

    @classmethod
    def struct_module(
        cls,
        softmax: nn.Module,
        dim: int = -1,
        in_act_bit: int = 8,
        softmax_act_bit: int = 16,
        out_act_bit: int = 8,
        act_quant_mode: str = "sym",
        act_per_token: bool = False,
        act_per_head: bool = False,
    ):
        return cls(
            dim=dim,
            in_act_bit=in_act_bit,
            softmax_act_bit=softmax_act_bit,
            out_act_bit=out_act_bit,
            act_quant_mode=act_quant_mode,
            act_per_token=act_per_token,
            act_per_head=act_per_head,
        ).copy_from(softmax)

# ----- IntLayerNorm -----

class IntLayerNorm(QuantizableModule):
    input_scale: Any
    output_scale: Any
    layernorm_scale: Any
    bias_scale: Any
    fused_scale: Any
    input_zero: Any
    output_zero: Any
    bias_int: Any

    def __init__(self, 
            normalized_shape, 
            eps=1e-5,
            elementwise_affine=True,
            bias=True,
            in_act_bit: int = 8,
            out_act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_channel: bool = False,
            input_act_per_token: bool = False,
            output_act_per_channel=None,
            int_cal_mode: str = "I-ViT",
        ):
        super(IntLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.int_cal_mode = int_cal_mode
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.in_act_bit = in_act_bit
        self.out_act_bit = out_act_bit
        self.act_quant_mode = act_quant_mode
        # Backward-compatible behavior:
        # - act_per_channel (legacy) maps to output_act_per_channel if the new flag isn't provided.
        if output_act_per_channel is None:
            output_act_per_channel = act_per_channel

        self.input_act_per_token = bool(input_act_per_token)
        self.output_act_per_channel = bool(output_act_per_channel)
        assert len(normalized_shape) == 1
        self.hidden_dim = normalized_shape[-1] if len(normalized_shape) == 1 else None # if you use it in Transformer this will be set
        self.dim_sqrt = torch.sqrt(torch.Tensor([self.hidden_dim]))

        # Input quant params can be per-token (dynamic length). Use placeholder and resize in calibrate().
        input_quant_param_shape = (1,)

        # Output quant params can be per-channel on hidden dim (static).
        if self.output_act_per_channel:
            output_quant_param_shape = tuple(normalized_shape)
        else:
            output_quant_param_shape = (1,)
        
        layernorm_quant_param_shape = normalized_shape

        # weight is fused into the output_scale
        # bias are always sym-quant
        # bias_scale == self.dim_sqrt / 2 ** 30
        if self.bias is not None:
            self.register_buffer('bias_scale', torch.zeros((1,)))
            # (x - mu) / std will offset the scale of input
            # bias_scale = dim_sqrt / 2 ** 30 that doesn't contain the scale of input
            # bias_int = bias / weight / bias_scale
            self.register_buffer('bias_int', torch.zeros_like(self.bias))
        else:
            self.register_buffer('bias_scale', None)
            self.register_buffer('bias_int', None)

        self.register_buffer('input_scale', torch.zeros(input_quant_param_shape))
        # output_scale comes from statistics
        self.register_buffer('output_scale', torch.zeros(output_quant_param_shape))
        # layernorm_scale == weight * bias_scale
        self.register_buffer('layernorm_scale', torch.zeros(layernorm_quant_param_shape))

        if act_quant_mode == 'sym':
            self.register_buffer('input_zero', None)
            self.register_buffer('output_zero', None)
            # output doesn't need zero for it's indeed 32bit int
        elif act_quant_mode == 'asym':
            self.register_buffer('input_zero', torch.zeros(input_quant_param_shape))
            self.register_buffer('output_zero', torch.zeros(output_quant_param_shape))
        else:
            raise NotImplementedError("unsupported act quant mode: {}".format(act_quant_mode))
        
        if len(normalized_shape) == 1:
            self.register_buffer("fused_scale", torch.zeros(layernorm_quant_param_shape))
            # self.qact = QAct(
            #     in_act_bit=32,
            #     out_act_bit=out_act_bit,
            #     in_act_quant_mode="sym", # int32 is not necessary to use asym
            #     out_act_quant_mode=act_quant_mode,
            #     in_per_channel=True,
            #     channel=self.hidden_dim,
            # )
        else:
            raise NotImplementedError("cannot support too complex qact per-token settings")

    def calibrate(self, x_float):
        y = self.forward_float(x_float)

        x_scale, x_zero = max_min_quantize_params(
            input_tensor=x_float,
            bitwidth=self.in_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.input_act_per_token,
            channel_dim=1 if self.input_act_per_token else None,
            is_weight=False,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.out_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.output_act_per_channel,
            channel_dim=2 if self.output_act_per_channel else None, # NOTE：注意这里要能够真的per-channel而非per-token，故3d的输入应该是actual_dim=2，需要使用manual channel_dim
            is_weight=False,
        )

        self.input_scale = x_scale
        self.output_scale = y_scale
        # TODO: update bias_scale algorithm to adapt to backend not this forward_int
        if self.bias is not None:
            if self.int_cal_mode == "I-ViT":
                self.bias_scale = self.dim_sqrt / 2 ** 30
                self.layernorm_scale = self.bias_scale * self.weight

                self.bias_int = symmetric_linear_quantize(
                    bits=32, # NOTE: hardcoded for now
                    input=self.bias / self.weight,
                    scale=self.bias_scale,
                    is_weight=True,
                )
            elif self.int_cal_mode == "Vivado-PYNQ":
                self.bias_scale = torch.Tensor([1 / (2 ** 16)]) # NOTE: hardcoded for now, please refer to pynq config
                self.layernorm_scale = self.bias_scale * self.weight

                self.bias_int = symmetric_linear_quantize(
                    bits=24, # NOTE: hardcoded for now, 24bit for PYNQ backend request
                    input=self.bias / self.weight,
                    scale=self.bias_scale,
                    is_weight=True,
                )
            else:
                raise NotImplementedError("unsupported int_cal_mode: {}".format(self.int_cal_mode))

        # self.qact.fused_scale.data = self.layernorm_scale / self.output_scale
        if self.int_cal_mode == "I-ViT":
            self.fused_scale = self.layernorm_scale / self.output_scale
        elif self.int_cal_mode == "Vivado-PYNQ":
            self.fused_scale = self.weight / self.output_scale
        else:
            raise NotImplementedError("unsupported int_cal_mode: {}".format(self.int_cal_mode))

        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.input_zero = x_zero
            self.output_zero = y_zero
            # NOTICE: self.qact.output_zero is not used
            # self.qact.output_zero.data = y_zero

        return y

    def forward_int(self, x_int):
        # Normalization: computes mean and variance(std)
        if self.int_cal_mode == "I-ViT":
            pass
        elif self.int_cal_mode == "Vivado-PYNQ":
            return x_int # NOTE: just a placeholder for Vivado-PYNQ to keep the shape while ShapeProp
            # layernorm doesn't change the shape
            # TODO: you can realize the truly int fake quant calculation for vivado pynq later here
        else:
            raise NotImplementedError("unsupported int_cal_mode: {}".format(self.int_cal_mode))

        if self.act_quant_mode == "asym":
            x_int = x_int - self.input_zero

        mean_int = torch.round(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 12
        for _ in range(10):
            k_1 = torch.floor((k + torch.floor(var_int/k))/2)
            k = k_1
        std_int = k

        factor = torch.floor(2 ** 30 / std_int)
        y_int = torch.floor(y_int * factor)

        if self.bias_int is not None:
            y_int = y_int + self.bias_int

        # y_int = self.qact.forward_int(y_int)
        y_int = y_int * self.fused_scale[None, None, :]
        # print(y_int,y_int*self.qact.fused_scale,self.qact.fused_scale,sep="\n")

        # Dequant zero
        if self.act_quant_mode == "asym":
            y_int = y_int + self.output_zero

        return y_int
    
    def forward_int_lut(self, x_int):
        """
        Integer LayerNorm using LUT.
        Forward path: i8 -> dequantize -> normalize -> insqrt_lut -> quantize -> i8
        """
        _init_vector_luts()
        
        if self.act_quant_mode == "asym":
            x_int = x_int - self.input_zero

        x_i32 = x_int.to(torch.int32)
        rows = x_i32.shape[-1]
        inv_rows_q16 = int(2 ** 16 // rows)

        if self.int_cal_mode == "Vivado-PYNQ":
            sum_i64 = torch.sum(x_i32.to(torch.int64), dim=-1, keepdim=True)
            sumsq_i64 = torch.sum((x_i32.to(torch.int64) * x_i32.to(torch.int64)), dim=-1, keepdim=True)

            mean_i64 = torch.bitwise_right_shift(sum_i64 * inv_rows_q16, 16)
            mean_square_i64 = torch.bitwise_right_shift(sumsq_i64 * inv_rows_q16, 16)
            sq_mean_i64 = mean_i64 * mean_i64
            var_i64 = torch.clamp(mean_square_i64 - sq_mean_i64, min=0)

            centered_i32 = x_i32 - mean_i64.to(torch.int32)
            idx = torch.clamp(
                torch.bitwise_right_shift(var_i64, 3),
                min=0,
                max=len(_g_norm_insqrt_lut) - 1,
            ).to(torch.long)
            insqrt_q16 = _g_norm_insqrt_lut.to(torch.int32)[idx]
            y_q16 = centered_i32.to(torch.int64) * insqrt_q16.to(torch.int64)

            y_int_like = y_q16.to(torch.float32)
            if self.bias_int is not None:
                y_int_like = y_int_like + self.bias_int
        else:
            raise NotImplementedError("unsupported int_cal_mode for LUT inference: {}".format(self.int_cal_mode))
            # mean_i64 = torch.round(x_i32.to(torch.float32).mean(dim=-1, keepdim=True)).to(torch.int64)
            # centered_i32 = x_i32 - mean_i64.to(torch.int32)

            # # Keep I-ViT statistics compatible with forward_int: var = sum((x-mean)^2)
            # var_i64 = torch.sum(
            #     centered_i32.to(torch.int64) * centered_i32.to(torch.int64),
            #     dim=-1,
            #     keepdim=True,
            # )
            # idx = torch.clamp(torch.bitwise_right_shift(var_i64, 6), min=0, max=len(_g_norm_insqrt_lut) - 1).to(torch.long)
            # insqrt_q16 = _g_norm_insqrt_lut.to(torch.int32)[idx]
            # y_q16 = centered_i32.to(torch.int64) * insqrt_q16.to(torch.int64)

            # # Q16 -> Q30, matching 2^30/std style factor in I-ViT branch.
            # y_q30 = torch.bitwise_left_shift(y_q16, 14)
            # y_int_like = y_q30.to(torch.float32)
            # if self.bias_int is not None:
            #     y_int_like = y_int_like + self.bias_int

        out_scale = self.fused_scale[None, None, :]
        if self.int_cal_mode == "Vivado-PYNQ":
            out_scale = out_scale / (2 ** 16)

        y_int_lut = torch.round(y_int_like * out_scale)
        y_int_lut = torch.clamp(y_int_lut, -2**(self.out_act_bit-1), 2**(self.out_act_bit-1)-1)
        
        if self.act_quant_mode == "asym":
            y_int_lut = y_int_lut + self.output_zero
        
        return y_int_lut
    
    def forward_float(self, x_float):
        return F.layer_norm(
            input=x_float, 
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

    def forward(self, x_float):
        if self.calibrate_mode:
            return self.calibrate(x_float)
        elif self.fakequant_mode:
            x_int = torch.clamp(torch.round(x_float / self.input_scale[None, :, None]), -2**(self.in_act_bit-1), 2**(self.in_act_bit-1)-1)
            y_int = self.forward_int_lut(x_int=x_int) if self.use_lut_inference else self.forward_int(x_int=x_int)
            return y_int * self.output_scale[None, None, :]
        else:
            return self.forward_float(x_float)
    
    def copy_from(self, layernorm):
        self.weight.data = layernorm.weight
        if layernorm.bias != None:
            self.bias.data = layernorm.bias
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, layernorm: nn.LayerNorm,
            in_act_bit: int = 8,
            out_act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_channel: bool = False,
            input_act_per_token: bool = False,
            output_act_per_channel=None,
            int_cal_mode: str = "I-ViT"):
        return cls(normalized_shape=layernorm.normalized_shape, 
            elementwise_affine=layernorm.elementwise_affine,
            bias=layernorm.bias != None,
            in_act_bit=in_act_bit,
            out_act_bit=out_act_bit,
            act_quant_mode=act_quant_mode,
            act_per_channel=act_per_channel,
            input_act_per_token=input_act_per_token,
            output_act_per_channel=output_act_per_channel,
            int_cal_mode=int_cal_mode,).copy_from(layernorm)

# ----- QAdd -----

# attach, not independent
class QAct(nn.Module):
    fused_scale: Any
    input_zero: Any
    output_zero: Any

    '''
    QAct的设计初衷应当是
    1.把所有可能的per_channel的都转成per_tensor
    2.把所有非标准量化bit（一般假设为8bit）的转为标准的统一
    3.如果全盘非对称设置，而输入是int32这种默认不引入zero的，那就也在这里转换
    所以很多冗余功能就不设计了，而且不是这两种情况也无需
    上面提到的对称，只要对称非对称全盘统一即可，标准量化8bit下要么全对称要么全不对称，
    '''
    def __init__(
            self, 
            in_act_bit: int = 8,
            out_act_bit: int = 8,
            in_act_quant_mode: str = "sym",
            out_act_quant_mode: str = "sym",
            in_per_token: bool = False,
            seqlen: int = 0,
        ):
        super(QAct, self).__init__()
        self.in_act_bit = in_act_bit
        self.out_act_bit = out_act_bit
        self.in_act_quant_mode = in_act_quant_mode
        self.out_act_quant_mode = out_act_quant_mode
        self.in_per_token = in_per_token

        if in_per_token:
            fused_quant_param_shape = (seqlen,)
        else:
            fused_quant_param_shape = (1,)

        # fused_scale = in_scale / out_scale
        self.register_buffer("fused_scale", torch.zeros(fused_quant_param_shape))
        if in_act_quant_mode == "sym":
            self.register_buffer("input_zero", None)
        elif in_act_quant_mode == "asym":
            self.register_buffer("input_zero", torch.zeros(fused_quant_param_shape))
        else:
            raise NotImplementedError("unsupported act quant mode: {}".format(in_act_quant_mode))
        
        if out_act_quant_mode == "sym":
            self.register_buffer("output_zero", None)
        elif out_act_quant_mode == "asym":
            self.register_buffer("output_zero", torch.zeros((1,)))
        else:
            raise NotImplementedError("unsupported act quant mode: {}".format(out_act_quant_mode))
        
    def forward_int(self, x_int):
        if self.in_act_quant_mode == "asym":
            x_int = x_int - self.input_zero
        
        y_int = torch.round(x_int * self.fused_scale[None, :, None])

        if self.out_act_quant_mode == "asym":
            y_int = y_int + self.output_zero
        return y_int
    
    def forward_float(self, x_float):
        return x_float

    def forward(self, x_float):
        return self.forward_float(x_float)

class QAdd(QuantizableModule):
    x_scale: Any
    y_scale: Any
    o_scale: Any
    x_fused_scale: Any
    y_fused_scale: Any
    x_zero: Any
    y_zero: Any
    o_zero: Any

    def __init__(
            self, 
            act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,
        ):
        super(QAdd, self).__init__()
        self.act_bit = act_bit
        self.act_quant_mode = act_quant_mode
        self.act_per_token = act_per_token

        if act_per_token:
            x_quant_param_shape = (1,)
            y_quant_param_shape = (1,)
            o_quant_param_shape = (1,)
            fused_scale_quant_param_shape = (1,)
        else: # per tensor
            x_quant_param_shape = (1,)
            y_quant_param_shape = (1,)
            o_quant_param_shape = (1,)
            fused_scale_quant_param_shape = (1,)
        
        self.register_buffer("x_scale", torch.zeros(x_quant_param_shape))
        self.register_buffer("y_scale", torch.zeros(y_quant_param_shape))
        self.register_buffer("o_scale", torch.zeros(o_quant_param_shape))
        # x_fused_scale == x_scale / o_scale
        self.register_buffer("x_fused_scale", torch.zeros(fused_scale_quant_param_shape))
        # y_fused_scale == y_scale / o_scale
        self.register_buffer("y_fused_scale", torch.zeros(fused_scale_quant_param_shape))

        if act_quant_mode == "sym":
            self.register_buffer("x_zero", None)
            self.register_buffer("y_zero", None)
            self.register_buffer("o_zero", None)
        elif act_quant_mode == "asym":
            self.register_buffer("x_zero", torch.zeros(x_quant_param_shape))
            self.register_buffer("y_zero", torch.zeros(y_quant_param_shape))
            self.register_buffer("o_zero", torch.zeros(o_quant_param_shape))

    def calibrate(self, x1_float, x2_float):
        y = self.forward_float(x1_float, x2_float)

        x1_scale, x1_zero = max_min_quantize_params(
            input_tensor=x1_float,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token,
            is_weight=False,
        )
        x2_scale, x2_zero = max_min_quantize_params(
            input_tensor=x2_float,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token,
            is_weight=False,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token,
            is_weight=False,
        )

        self.x_scale.data = x1_scale
        self.y_scale.data = x2_scale
        self.o_scale.data = y_scale
        self.x_fused_scale.data = x1_scale / y_scale
        self.y_fused_scale.data = x2_scale / y_scale

        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.x_zero.data = x1_zero
            self.y_zero.data = x2_zero
            self.o_zero.data = y_zero

        return y

    def forward_int(self, x_int, y_int):
        if self.act_quant_mode == "asym":
            x_int = x_int - self.x_zero
            y_int = y_int - self.y_zero
        
        o_int = torch.round(x_int * 2**8 * self.x_fused_scale[None, :, None] + y_int * 2**8 * self.y_fused_scale[None, :, None])
        o_int = o_int / 2**8

        if self.act_quant_mode == "asym":
            o_int = o_int + self.o_zero

        return o_int
    
    def forward_int_lut(self, x_int, y_int):
        """
        Integer QAdd with pseudo-quantized arithmetic.
        Forward path: i8 + i8 -> dequantize -> add -> quantize -> i8
        """
        return self.forward_int(x_int, y_int)

    def forward_float(self, x_float, y_float):
        return x_float + y_float
    
    def forward(self, x_float, y_float):
        if self.calibrate_mode:
            return self.calibrate(x_float, y_float)
        elif self.fakequant_mode:
            # TODO: think of add the zero of asym quant here, the others are the same!!!
            # TODO: think of add the zero of asym quant here
            # TODO: think of add the zero of asym quant here
            x_int = torch.clamp(torch.round(x_float / self.x_scale[None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)
            y_int = torch.clamp(torch.round(y_float / self.y_scale[None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)
            o_int = self.forward_int_lut(x_int=x_int, y_int=y_int) if self.use_lut_inference else self.forward_int(x_int=x_int, y_int=y_int)
            return o_int * self.o_scale[None, :, None]
        else:
            return self.forward_float(x_float, y_float)

    def copy_from(self, add):
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, add: Add, 
            act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,):
        return cls(act_bit=act_bit, act_quant_mode=act_quant_mode, act_per_token=act_per_token).copy_from(add)

class QMatMul(QuantizableModule):
    x_scale: Any
    y_scale: Any
    o_scale: Any
    fused_scale: Any
    x_zero: Any
    y_zero: Any
    o_zero: Any

    def __init__(
            self,
            act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,
        ):
        super(QMatMul, self).__init__()
        self.act_bit = act_bit
        self.act_quant_mode = act_quant_mode
        self.act_per_token = act_per_token # just affect on x1 or input1 and output

        if act_per_token:
            x_quant_param_shape = (1,)
            y_quant_param_shape = (1,)
            o_quant_param_shape = (1,)
            fused_scale_quant_param_shape = (1,)
        else: # per tensor
            x_quant_param_shape = (1,)
            y_quant_param_shape = (1,)
            o_quant_param_shape = (1,)
            fused_scale_quant_param_shape = (1,)
        
        self.register_buffer("x_scale", torch.zeros(x_quant_param_shape))
        self.register_buffer("y_scale", torch.zeros(y_quant_param_shape))
        self.register_buffer("o_scale", torch.zeros(o_quant_param_shape))
        # fused_scale == x_scale * y_scale / o_scale
        self.register_buffer("fused_scale", torch.zeros(fused_scale_quant_param_shape))

        if act_quant_mode == "sym":
            self.register_buffer("x_zero", None)
            self.register_buffer("y_zero", None)
            self.register_buffer("o_zero", None)
        elif act_quant_mode == "asym":
            self.register_buffer("x_zero", torch.zeros(x_quant_param_shape))
            self.register_buffer("y_zero", torch.zeros(y_quant_param_shape))
            self.register_buffer("o_zero", torch.zeros(o_quant_param_shape))

    def calibrate(self, x1_float, x2_float):
        # if it's ISqrtD version, the forward_float will point to the children
        y = self.forward_float(x1_float, x2_float)

        x1_scale, x1_zero = max_min_quantize_params(
            input_tensor=x1_float,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token,
            is_weight=False,
            is_seq_x=self.act_per_token, # per-channel must be True to use this, act_per_token means it reduces on dim=2, otherwise is act_per_head which reduces on dim = 1
        )
        x2_scale, x2_zero = max_min_quantize_params(
            input_tensor=x2_float,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=False, # Matmul not quant perchannel on x2
            is_weight=False,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=self.act_per_token,
            is_weight=False,
            is_seq_x=self.act_per_token, 
        )

        self.x_scale.data = x1_scale
        self.y_scale.data = x2_scale
        self.o_scale.data = y_scale
        # TODO: if per_channel?
        self.fused_scale.data = x1_scale * x2_scale / y_scale

        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.x_zero.data = x1_zero
            self.y_zero.data = x2_zero
            self.o_zero.data = y_zero

        return y

    def forward_int(self, x_int, y_int):
        if self.act_quant_mode == "asym":
            x_int = x_int - self.x_zero
            y_int = y_int - self.y_zero

        o_int = torch.round((x_int @ y_int) * self.fused_scale[None, None, :, None])

        if self.act_quant_mode == "asym":
            o_int = o_int + self.o_zero

        return o_int
    
    def forward_int_lut(self, x_int, y_int):
        """
        Integer QMatMul with pseudo-quantized arithmetic.
        Forward path: i8 @ i8 -> dequantize -> matmul -> quantize -> i8
        """
        if self.act_quant_mode == "asym":
            x_int = x_int - self.x_zero
            y_int = y_int - self.y_zero

        acc = x_int @ y_int
        drop_bits = 8
        acc_drop = torch.bitwise_right_shift(acc.to(torch.int64), drop_bits).to(torch.float32)
        fused = self.fused_scale[None, None, :, None] * (2 ** drop_bits)
        o_int_lut = torch.round(acc_drop * fused)
        o_int_lut = torch.clamp(o_int_lut, -2**(self.act_bit-1), 2**(self.act_bit-1)-1)

        if self.act_quant_mode == "asym":
            o_int_lut = o_int_lut + self.o_zero

        return o_int_lut

    def forward_float(self, x_float, y_float):
        return torch.matmul(x_float, y_float)
    
    def forward(self, x_float, y_float):
        if self.calibrate_mode:
            return self.calibrate(x_float, y_float)
        elif self.fakequant_mode:
            x_int = torch.clamp(torch.round(x_float / self.x_scale[None, None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)
            y_int = torch.clamp(torch.round(y_float / self.y_scale), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)
            o_int = self.forward_int_lut(x_int=x_int, y_int=y_int) if self.use_lut_inference else self.forward_int(x_int=x_int, y_int=y_int)
            return o_int * self.o_scale[None, None, :, None]
        else:
            return self.forward_float(x_float, y_float)

    def copy_from(self, matmul):
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, matmul: MatMul,
            act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,):
        return cls(act_bit=act_bit,
            act_quant_mode=act_quant_mode,
            act_per_token=act_per_token,).copy_from(matmul)

class QMatMulIsqrtD(QMatMul):
    def __init__(
        self,
        dim: int,
        act_bit: int = 8,
        act_quant_mode: str = "sym",
        act_per_token: bool = False,
    ):
        super(QMatMulIsqrtD, self).__init__(
            act_bit=act_bit,
            act_quant_mode=act_quant_mode,
            act_per_token=act_per_token,
        )
        self.sqrt_dim = torch.sqrt(torch.Tensor([dim]))
    #     self.enable_row_gain_calibration = str(os.getenv("ALLO_QK_ROW_GAIN_CALIB", "1")).lower() not in (
    #         "0", "false", "off"
    #     )
    #     self.row_gain_clip = float(os.getenv("ALLO_QK_ROW_GAIN_CLIP", "0.15"))
    #     self.row_gain_eps = float(os.getenv("ALLO_QK_ROW_GAIN_EPS", "1e-8"))

    # def _estimate_row_gain_from_lut(self, x1_float, x2_float, y_float_ref):
    #     # Only apply to qk-like 4D path with per-token fused scales.
    #     if x1_float.ndim != 4 or x2_float.ndim != 4:
    #         return None
    #     if self.fused_scale.numel() <= 1:
    #         return None

    #     # Quantize using calibrated scales (same formula as forward fakequant path).
    #     x_int = torch.clamp(
    #         torch.round(x1_float / self.x_scale[None, None, :, None]),
    #         -2 ** (self.act_bit - 1),
    #         2 ** (self.act_bit - 1) - 1,
    #     )
    #     y_int = torch.clamp(
    #         torch.round(x2_float / self.y_scale),
    #         -2 ** (self.act_bit - 1),
    #         2 ** (self.act_bit - 1) - 1,
    #     )

    #     # Match current LUT integer path (drop_bits=8) to estimate deploy-time bias.
    #     acc = x_int @ y_int
    #     drop_bits = 8
    #     acc_drop = torch.bitwise_right_shift(acc.to(torch.int64), drop_bits).to(torch.float32)
    #     fused = self.fused_scale[None, None, :, None] * (2 ** drop_bits)
    #     o_int_lut = torch.round(acc_drop * fused)
    #     o_int_lut = torch.clamp(o_int_lut, -2 ** (self.act_bit - 1), 2 ** (self.act_bit - 1) - 1)

    #     y_lut = o_int_lut * self.o_scale[None, None, :, None]

    #     # Per-token least-squares slope: alpha_t = <y_ref, y_lut> / <y_lut, y_lut>
    #     num = torch.sum(y_float_ref * y_lut, dim=(0, 1, 3))
    #     den = torch.sum(y_lut * y_lut, dim=(0, 1, 3)) + self.row_gain_eps
    #     alpha = num / den

    #     clip = max(0.0, float(self.row_gain_clip))
    #     alpha = torch.clamp(alpha, 1.0 - clip, 1.0 + clip)
    #     return alpha

    def forward_float(self, x_float, y_float):
        return super().forward_float(x_float, y_float) / self.sqrt_dim

    def calibrate(self, x1_float, x2_float):
        # Keep QMatMul generic: fuse sqrt(dim) only for this variant.
        y = super().calibrate(x1_float, x2_float)

        # fused_scale == x_scale * y_scale / o_scale, and for ISqrtD we need an
        # extra 1/sqrt(dim) so that dequant restores (x@y)/sqrt(dim).
        self.fused_scale.data = self.fused_scale / self.sqrt_dim

        # print(self.o_scale)

        # if self.enable_row_gain_calibration and self.act_quant_mode == "sym":
        #     alpha = self._estimate_row_gain_from_lut(x1_float, x2_float, y)
        #     if alpha is not None:
        #         self.fused_scale.data = self.fused_scale * alpha

        return y
    
    def forward_int(self, x_int, y_int):
        # NOTICE: Here we fuse the sqrt_dim into output_scale and cofused_scalee
        # But this request you to get the correct fused_scale and scale first
        return super().forward_int(x_int, y_int)
    
    def forward_int_lut(self, x_int, y_int):
        """
        Integer QMatMulIsqrtD with division by sqrt(dim).
        Forward path: i8 @ i8 / sqrt(dim) -> quantized scaling
        """
        return super().forward_int_lut(x_int, y_int)
    
    def copy_from(self, matmul):
        self.sqrt_dim = matmul.sqrt_dim
        return self

    # TODO: can we pass bit setup?
    @classmethod
    def struct_module(cls, matmul: MatMulIsqrtD,
            act_bit: int = 8,
            act_quant_mode: str = "sym",
            act_per_token: bool = False,):
        return cls(dim=matmul.sqrt_dim**2,
            act_bit=act_bit,
            act_quant_mode=act_quant_mode,
            act_per_token=act_per_token,).copy_from(matmul)

### TEST ###

def test_gelu():
    # test gelu
    batch = 64
    dim = 192
    bit = 8
    int_gelu = IntGELU()

    x = torch.randn((1, dim)).clamp(-1, 1)
    samples = torch.randn((batch, dim)).clamp(-1, 1)
    int_gelu.start_calibrate()
    int_gelu(samples)
    int_gelu.stop_calibrate()
    x_int = symmetric_linear_quantize(
        bits=int_gelu.act_bit, 
        input=x, 
        scale=int_gelu.input_scale, 
        is_weight=False,
    )
    y_int = int_gelu.forward_int(x_int)
    y = int_gelu(x)
    print("The max difference:", torch.max(torch.abs(y - y_int * int_gelu.output_scale.data)))
    print(y, y_int * int_gelu.output_scale.data, y_int, sep="\n")

def test_softmax():
    # test softmax
    dim = 1024
    len = 32
    batch = 64
    ibit, obit = 8, 16

    int_softmax = IntSoftmax(in_act_bit=ibit, out_act_bit=obit)

    x = torch.randn((len, dim)).clamp(-1, 1)
    samples = torch.randn((batch, len, dim)).clamp(-1, 1)

    int_softmax.start_calibrate()
    int_softmax(samples)
    int_softmax.stop_calibrate()

    x_int = symmetric_linear_quantize(
        bits=int_softmax.in_act_bit,
        input=x,
        scale=int_softmax.input_scale,
        is_weight=False,
    )

    y_int = int_softmax.forward_int(x_int)
    y = int_softmax(x)
    print(y, y_int, y_int * int_softmax.output_scale, sep="\n")
    print(y.sum(dim=-1), (y_int * int_softmax.output_scale).sum(dim=-1), sep="\n")
    # print(torch.exp(x), int_softmax._int_exp(x_int, int_softmax.input_scale) * int_softmax.input_scale / 2 ** (int_softmax.n), sep="\n")

def test_layernorm():
    # test layernorm
    batch, len, dim = 64, 128, 192
    ibit, obit = 8, 8
    int_layernorm = IntLayerNorm(
        normalized_shape=[dim],
        in_act_bit=ibit, 
        out_act_bit=obit
    )
    int_layernorm.eval()
    # x = torch.randn((1, len, dim)).clamp(-1, 1)
    # x[0, :, 0] = 0.0002
    # x[0, :, -1] = 9.0001
    samples = torch.randn((batch, len, dim)).clamp(-10, 10)
    x = torch.tensor(np.load("ln_f.npy"))
    # samples = x
    w = torch.randn((dim,)).clamp(-1, 1)
    b = torch.randn((dim,)).clamp(-1, 1)

    int_layernorm.start_calibrate()
    int_layernorm(samples)
    int_layernorm.stop_calibrate()

    x_int = symmetric_linear_quantize(
        bits=int_layernorm.in_act_bit,
        input=x,
        scale=int_layernorm.input_scale,
        is_weight=False,
    )

    y_int = int_layernorm.forward_int(x_int)
    y = int_layernorm(x)
    print(y, y_int, y_int * int_layernorm.output_scale.data, sep="\n")
    print(f"mean diff: {torch.mean(y - y_int * int_layernorm.output_scale.data)}")
    print(f"max diff: {torch.max(y - y_int * int_layernorm.output_scale.data)}")

def test_linear():
    # test linear
    batch = 64
    ift, oft = 192, 32
    bit = 8
    qlinear = QLinear(ift, oft)
    w = torch.randn((oft, ift)) * 0.02
    b = torch.randn((oft,)).clamp(-0.5, 0.5)
    x = torch.randn((1, ift)).clamp(-1, 1)

    samples = torch.randn((batch, ift)).clamp(-1, 1)
    qlinear.start_calibrate()
    qlinear(samples)
    qlinear.stop_calibrate()
    x_int = symmetric_linear_quantize(
        bits=qlinear.act_bit, 
        input=x, 
        scale=qlinear.input_scale, 
        is_weight=False,
    )
    y_int = qlinear.forward_int(x_int)
    y = qlinear(x)
    print("The max difference:", torch.max(torch.abs(y - y_int * qlinear.output_scale.data)))
    print(y, y_int * qlinear.output_scale.data, y_int, sep="\n")

def test_conv2d():
    # test conv2d
    batch = 64
    ift, oft = 64, 16
    bit = 8
    qconv2d = QConv2d(ift, oft, 2, 2, 0, act_bit=bit)
    w = torch.randn((oft, ift, 2, 2)) * 0.02
    b = torch.randn((oft,)).clamp(-0.5, 0.5)
    x = torch.randn((ift, 4, 4)).clamp(-1, 1)
    samples = torch.randn((batch, ift, 4, 4)).clamp(-1, 1)
    qconv2d.start_calibrate()
    qconv2d(samples)
    qconv2d.stop_calibrate()
    x_int = symmetric_linear_quantize(
        bits=qconv2d.act_bit, 
        input=x, 
        scale=qconv2d.input_scale, 
        is_weight=False,
    )
    y_int = qconv2d.forward_int(x_int)
    y = qconv2d(x)
    print("The max difference:", torch.max(torch.abs(y - y_int * qconv2d.output_scale.data)))
    print(y, y_int * qconv2d.output_scale.data, y_int, sep="\n")

def test_qmatmul():
    bit = 8
    M, N, K = 16, 32, 64
    qmatmul = QMatMulIsqrtD(dim=K, act_bit=bit)

    X = torch.randn((M, K)).clamp(-1, 1)
    Y = torch.randn((K, N)).clamp(-1, 1)

    X_sample = torch.randn((64, M, K)).clamp(-1, 1)
    Y_sample = torch.randn((64, K, N)).clamp(-1, 1)

    qmatmul.start_calibrate()
    qmatmul(X_sample, Y_sample)
    qmatmul.stop_calibrate()

    X_int = symmetric_linear_quantize(
        bits=qmatmul.act_bit,
        input=X,
        scale=qmatmul.x_scale,
        is_weight=False,
    )
    Y_int = symmetric_linear_quantize(
        bits=qmatmul.act_bit,   
        input=Y,
        scale=qmatmul.y_scale,
        is_weight=False,
    )

    z_int = qmatmul.forward_int(x_int=X_int, y_int=Y_int)
    z = qmatmul(X, Y)
    print(z, z_int, z_int * qmatmul.o_scale.data, sep="\n")

def test_qadd():
    bit = 8
    qadd = QAdd(act_bit=bit)

    X = torch.randn((16,8)).clamp(-2, 2)
    Y = torch.randn((16,8)).clamp(-1, 1)
    X_sample = torch.randn((64,16,8)).clamp(-2, 2)
    Y_sample = torch.randn((64,16,8)).clamp(-1, 1)

    qadd.start_calibrate()
    qadd(X_sample, Y_sample)
    qadd.stop_calibrate()

    X_int = symmetric_linear_quantize(
        bits=qadd.act_bit,
        input=X,
        scale=qadd.x_scale,
        is_weight=False,
    )
    Y_int = symmetric_linear_quantize(
        bits=qadd.act_bit,
        input=Y,
        scale=qadd.y_scale,
        is_weight=False,
    )

    z_int = qadd.forward_int(x_int=X_int, y_int=Y_int)
    z = qadd(X, Y)
    print(z, z_int, z_int * qadd.o_scale.data, sep="\n")

if __name__ == "__main__":
    # test_gelu()
    # test_linear()
    # test_conv2d()
    # test_qadd()
    # test_qmatmul()
    test_layernorm()
    # test_softmax()
