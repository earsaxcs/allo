import allo
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .quant_utils import *
from typing import Any
from ..ops.vit import Add, MatMul, MatMulIsqrtD

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

    def stop_calibrate(self):
        self.calibrate_mode = False

    def start_calibrate(self):
        self.calibrate_mode = True

    def disable_fakequant(self):
        self.fakequant_mode = False

    def enable_fakequant(self):
        self.fakequant_mode = True

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
                 bias_bit: int = 32,
                 act_bit: int = 8,
                 act_quant_mode: str = "sym",
                 act_per_token: bool = False,
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
        self.act_per_token = act_per_token
        self.wgt_per_channel = wgt_per_channel

        # can't be used at the same time
        assert not(wgt_per_channel and act_per_token) and "pt and pc can't be used at the same time"

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
        if self.act_per_token:
            input_quant_param_shape = (1,)
            output_quant_param_shape = (1,)
        else: # per tensor
        # NOTICE: we can't assume that output_scale == bias_scale == input_scale * weight_scale, because this will make the output be a int32 number
            input_quant_param_shape = (1,)
            output_quant_param_shape = (1,)
        fused_scale_quant_param_shape = (1 * input_quant_param_shape[0] * self.bias_scale.numel(),)

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

        # no bias_zero
        # NOTICE: since perchannel and pertoken would not be used at the same time, here is secure
        b_scale = x_scale * w_scale

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
        if self.bias is not None:
            # NOTICE: No need to add bias if no bias
            # but bias_scale can exist even there is no bias
            # TODO: !!!!!!!!!!!!!!bias_bit must 32???
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

    def forward_int(self, x_int):
        if self.act_quant_mode == "asym":
            # asymmetric
            x_int = x_int - self.input_zero

        if self.wgt_per_channel:
            fused_scale = self.fused_scale[None, :]
        elif self.act_per_token:
            fused_scale = self.fused_scale[:, None]
        else:
            fused_scale = self.fused_scale

        o_int = F.linear(x_int, self.weight_int, self.bias_int)
        o_int = torch.round(o_int * fused_scale) # automatically broadcast Batchsize and Channel/Token
        
        if self.act_quant_mode == "asym":
            o_int = o_int + self.output_zero

        return o_int
    
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
            # TODO: think of asymmetric
            return self.forward_int(x_int=torch.clamp(torch.round(x_float / self.input_scale[:, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)) * self.output_scale[:, None]
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
                 wgt_per_channel=False):
        return cls(linear.in_features, 
                   linear.out_features,
                   weight_bit=weight_bit,
                   bias_bit=bias_bit,
                   act_bit=act_bit,
                   act_quant_mode=act_quant_mode,
                   act_per_token=act_per_token,
                   wgt_per_channel=wgt_per_channel,
                   ).copy_from(linear)

# ----- QConv2d -----

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
    
    def forward(self, x_float):
        if self.calibrate_mode:
            return self.calibrate(x_float)
        elif self.fakequant_mode:
            return self.forward_int(x_int=torch.clamp(torch.round(x_float / self.input_scale), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)) * self.output_scale[:, None, None]
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

    def forward_int(self, x_int):
        x_int = self.fc1.forward_int(x_int)
        x_int = self.activation.forward_int(x_int)
        x_int = self.fc2.forward_int(x_int)
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

    def forward(self, x_float):
        if self.calibrate_mode:
            return self.calibrate(x_float)
        elif self.fakequant_mode:
            return self.forward_int(x_int=torch.clamp(torch.round(x_float / self.input_scale[None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)) * self.output_scale[None, :, None]
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
            self.qact.output_zero.data = y_zero

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

    def forward(self, x_float):
        if self.calibrate_mode:
            return self.calibrate(x_float)
        elif self.fakequant_mode:
            self.debug_count += 1
            tmp = self.forward_int(x_int=torch.clamp(torch.round(x_float / self.input_scale[None, None, :, None]), -2**(self.in_act_bit-1), 2**(self.in_act_bit-1)-1)) * self.output_scale[None, None, :, None]
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
        ):
        super(IntLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
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
        self.act_per_channel = act_per_channel
        assert len(normalized_shape) == 1
        self.hidden_dim = normalized_shape[-1] if len(normalized_shape) == 1 else None # if you use it in Transformer this will be set
        self.dim_sqrt = torch.sqrt(torch.Tensor([self.hidden_dim]))

        # here just control the input quant param shape
        if act_per_channel:
            self.act_per_channel = act_per_channel
            input_quant_param_shape = tuple(normalized_shape)
            output_quant_param_shape = tuple(normalized_shape)
        else:
            input_quant_param_shape = (1,)
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
            self.register_buffer("fused_scale", torch.zeros(input_quant_param_shape))
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
            bitwidth=self.out_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=False, 
            channel_dim=2 if self.act_per_channel else None, # 注意这里要能够真的per-channel而非per-token，故3d的输入应该是actual_dim=2，需要使用manual channel_dim
            is_weight=False,
        )

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=self.out_act_bit,
            quant_mode=self.act_quant_mode,
            per_channel=False,
            channel_dim=2 if self.act_per_channel else None,
            is_weight=False,
        )

        self.input_scale.data = x_scale
        self.output_scale.data = y_scale
        if self.bias is not None:
            self.bias_scale.data = self.dim_sqrt / 2 ** 30
            self.layernorm_scale.data = self.bias_scale * self.weight

            self.bias_int.data = symmetric_linear_quantize(
                bits=32,
                input=self.bias / self.weight,
                scale=self.bias_scale,
                is_weight=True,
            )

        # self.qact.fused_scale.data = self.layernorm_scale / self.output_scale
        self.fused_scale.data = self.layernorm_scale / self.output_scale
        
        if self.act_quant_mode == "sym":
            pass
        elif self.act_quant_mode == "asym":
            self.input_zero.data = x_zero
            self.output_zero.data = y_zero
            # NOTICE: self.qact.output_zero is not used
            # self.qact.output_zero.data = y_zero

        return y

    def forward_int(self, x_int):
        # Normalization: computes mean and variance(std)
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
            return self.forward_int(x_int=torch.clamp(torch.round(x_float / self.input_scale[None, None, :]), -2**(self.in_act_bit-1), 2**(self.in_act_bit-1)-1)) * self.output_scale[None, None, :]
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
            act_per_channel: bool = False,):
        return cls(normalized_shape=layernorm.normalized_shape, 
            elementwise_affine=layernorm.elementwise_affine,
            bias=layernorm.bias != None,
            in_act_bit=in_act_bit,
            out_act_bit=out_act_bit,
            act_quant_mode=act_quant_mode,
            act_per_channel=act_per_channel).copy_from(layernorm)

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

    def forward_float(self, x_float, y_float):
        return x_float + y_float
    
    def forward(self, x_float, y_float):
        if self.calibrate_mode:
            return self.calibrate(x_float, y_float)
        elif self.fakequant_mode:
            # TODO: think of add the zero of asym quant here, the others are the same!!!
            # TODO: think of add the zero of asym quant here
            # TODO: think of add the zero of asym quant here
            return self.forward_int(x_int=torch.clamp(torch.round(x_float / self.x_scale[None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1), y_int=torch.clamp(torch.round(y_float / self.y_scale[None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)) * self.o_scale[None, :, None]
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

        # for it's children ISqrtD
        if hasattr(self, "sqrt_dim"):
            self.fused_scale.data = self.fused_scale / self.sqrt_dim

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

    def forward_float(self, x_float, y_float):
        return torch.matmul(x_float, y_float)
    
    def forward(self, x_float, y_float):
        if self.calibrate_mode:
            return self.calibrate(x_float, y_float)
        elif self.fakequant_mode:
            return self.forward_int(x_int=torch.clamp(torch.round(x_float / self.x_scale[None, None, :, None]), -2**(self.act_bit-1), 2**(self.act_bit-1)-1), y_int=torch.clamp(torch.round(y_float / self.y_scale), -2**(self.act_bit-1), 2**(self.act_bit-1)-1)) * self.o_scale[None, None, :, None]
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

    def forward_float(self, x_float, y_float):
        return super().forward_float(x_float, y_float) / self.sqrt_dim
    
    def forward_int(self, x_int, y_int):
        # NOTICE: Here we fuse the sqrt_dim into output_scale and cofused_scalee
        # But this request you to get the correct fused_scale and scale first
        return super().forward_int(x_int, y_int)
    
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
