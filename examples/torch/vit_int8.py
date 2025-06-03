import allo
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from quant_modules import *
from quant_utils import *

class Calibrator:
    def __init__(
        self,
        qmodel: nn.Module,
        example_inputs: torch.Tensor,
    ):
        self.qmodel = qmodel
        self.example_inputs = example_inputs

    def calibrate(self):
        self.calibrate_module(
            qmodel=self.qmodel,
            x=self.example_inputs,
            pre_scale=torch.Tensor([1]),
            pre_zero=torch.Tensor([0]),
        )
        
    def calibrate_module(
        self, 
        qmodel: nn.Module,
        x: torch.Tensor,
        pre_scale: torch.Tensor,
        pre_zero: torch.Tensor | None = None,
    ):
        for name, module in self.qmodel.named_modules():
            if isinstance(module, QLinear):
                self.calib_linear(module, self.example_inputs, pre_scale, pre_zero)

    def calib_linear(
            self, 
            qlinear: QLinear, 
            x: torch.Tensor, 
            pre_scale: torch.Tensor,
            pre_zero: torch.Tensor | None = None,
        ):
        y = qlinear(x)
        
        w_scale, _ = max_min_quantize_params(
            input_tensor=qlinear.weight.data,
            bitwidth=qlinear.weight_bit,
            quant_mode="sym", # weight强制sym
            per_channel=True, # weight先假设per_channel
            is_weight=True,
        )

        # x_scale, x_zero = max_min_quantize_params(
        #     input_tensor=x,
        #     bitwidth=qlinear.act_bit,
        #     quant_mode=qlinear.act_quant_mode,
        #     per_channel=qlinear.act_per_channel,
        #     is_weight=False,
        # )
        x_scale, x_zero = pre_scale, pre_zero

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=qlinear.act_bit,
            quant_mode=qlinear.act_quant_mode,
            per_channel=qlinear.act_per_channel,
            is_weight=False,
        )

        # no bias_zero
        b_scale = x_scale * w_scale

        # insert quant params
        qlinear.input_scale.data = x_scale
        qlinear.output_scale.data = y_scale
        qlinear.weight_scale.data = w_scale
        qlinear.bias_scale.data = b_scale
        qlinear.coe.data = b_scale / y_scale
        qlinear.weight_int.data = symmetric_linear_quantize(
            bits=qlinear.weight_bit, 
            input=qlinear.weight.data, 
            scale=w_scale, 
            is_weight=True,
        )
        if qlinear.bias is not None:
            # NOTICE: No need to add bias if no bias
            # but bias_scale can exist even there is no bias
            qlinear.bias_int.data = symmetric_linear_quantize(
                bits=qlinear.bias_bit, 
                input=qlinear.bias.data, 
                scale=b_scale, 
                is_weight=True,
            )

        if qlinear.act_quant_mode == "sym":
            pass
        elif qlinear.act_quant_mode == "asym":
            qlinear.input_zero.data = x_zero
            qlinear.output_zero.data = y_zero
        
        return y, y_scale, y_zero

    def calib_conv2d(self, 
            qconv2d: QConv2d, 
            x: torch.Tensor, 
            pre_scale: torch.Tensor,
            pre_zero: torch.Tensor | None = None,
        ):
        y = qconv2d(x)
        
        w_scale, _ = max_min_quantize_params(
            input_tensor=qconv2d.weight.data,
            bitwidth=qconv2d.weight_bit,
            quant_mode="sym", # weight强制sym
            per_channel=True,
            is_weight=True,
        )

        # x_scale, x_zero = max_min_quantize_params(
        #     input_tensor=x,
        #     bitwidth=qconv2d.act_bit,
        #     quant_mode=qconv2d.act_quant_mode,
        #     per_channel=qconv2d.act_per_channel,
        #     is_weight=False,
        # )
        x_scale, x_zero = pre_scale, pre_zero

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=qconv2d.act_bit,
            quant_mode=qconv2d.act_quant_mode,
            per_channel=qconv2d.act_per_channel,
            is_weight=False,
        )

        b_scale = x_scale * w_scale
        # no bias_zero

        # insert quant params
        qconv2d.input_scale.data = x_scale
        qconv2d.output_scale.data = y_scale
        qconv2d.weight_scale.data = w_scale
        qconv2d.bias_scale.data = b_scale
        qconv2d.coe.data = b_scale / y_scale
        qconv2d.weight_int.data = symmetric_linear_quantize(
            bits=qconv2d.weight_bit, 
            input=qconv2d.weight.data, 
            scale=w_scale, 
            is_weight=True,
        )
        qconv2d.bias_int.data = symmetric_linear_quantize(
            bits=qconv2d.bias_bit, 
            input=qconv2d.bias.data, 
            scale=b_scale, 
            is_weight=True,
        )
        if qconv2d.act_quant_mode == "sym":
            pass
        elif qconv2d.act_quant_mode == "asym":
            qconv2d.input_zero.data = x_zero
            qconv2d.output_zero.data = y_zero
        
        return y, y_scale, y_zero

    def calib_gelu(
        self,
        intgelu: IntGELU, 
        x: torch.Tensor, 
        pre_scale: torch.Tensor,
        pre_zero: torch.Tensor | None = None,
    ):
        y = intgelu(x)

        x_scale, x_zero = pre_scale, pre_zero

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=intgelu.act_bit,
            quant_mode=intgelu.act_quant_mode,
            per_channel=intgelu.act_per_channel,
            is_weight=False,
        )

        intgelu.input_scale.data = x_scale
        intgelu.output_scale.data = y_scale
        intgelu.gelu_scale.data = x_scale / 2 ** (intgelu.act_bit - 1)
        intgelu.coe.data = intgelu.gelu_scale / intgelu.output_scale

        if intgelu.act_quant_mode == "sym":
            pass
        elif intgelu.act_quant_mode == "asym":
            intgelu.input_zero.data = x_zero
            intgelu.output_zero.data = y_zero

        return y, y_scale, y_zero

    def calib_softmax(
        self,
        intsoftmax: IntSoftmax,
        x: torch.Tensor,
        pre_scale: torch.Tensor,
        pre_zero: torch.Tensor | None = None,
    ):
        y = intsoftmax(x)

        x_scale, x_zero = pre_scale, pre_zero

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=intsoftmax.out_act_bit,
            quant_mode=intsoftmax.act_quant_mode,
            per_channel=intsoftmax.act_per_channel,
            is_weight=False,
        )

        intsoftmax.input_scale.data = x_scale
        intsoftmax.output_scale.data = y_scale
        intsoftmax.softmax_scale.data = torch.Tensor([1 / 2 ** (intsoftmax.softmax_act_bit - 1)])
        if intsoftmax.softmax_act_bit != intsoftmax.out_act_bit:
            intsoftmax.qact.fused_scale.data = intsoftmax.softmax_scale / intsoftmax.output_scale
        else: # TODO: ??
            intsoftmax.output_scale.data = intsoftmax.softmax_scale

        if intsoftmax.act_quant_mode == "sym":
            pass
        elif intsoftmax.act_quant_mode == "asym":
            intsoftmax.input_zero.data = x_zero
            intsoftmax.output_zero.data = y_zero
            intsoftmax.qact.output_zero.data = y_zero

        return y, y_scale, y_zero
    
    def calib_layernorm(
        self,
        intlayernorm: IntLayerNorm,
        x: torch.Tensor,
        pre_scale: torch.Tensor,
        pre_zero: torch.Tensor | None = None,
    ):
        y = intlayernorm(x)

        x_scale, x_zero = pre_scale, pre_zero

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=intlayernorm.out_act_bit,
            quant_mode=intlayernorm.act_quant_mode,
            per_channel=intlayernorm.act_per_channel,
            is_weight=False,
        )

        intlayernorm.input_scale.data = x_scale
        intlayernorm.output_scale.data = y_scale
        intlayernorm.bias_scale.data = intlayernorm.dim_sqrt / 2 ** 30
        intlayernorm.layernorm_scale.data = intlayernorm.bias_scale * intlayernorm.weight

        intlayernorm.bias_int.data = symmetric_linear_quantize(
            bits=intlayernorm.out_act_bit,
            input=intlayernorm.bias / intlayernorm.weight,
            scale=intlayernorm.bias_scale,
            is_weight=True,
        )

        intlayernorm.qact.fused_scale.data = intlayernorm.layernorm_scale / intlayernorm.output_scale
        
        if intlayernorm.act_quant_mode == "sym":
            pass
        elif intlayernorm.act_quant_mode == "asym":
            intlayernorm.input_zero.data = x_zero
            intlayernorm.output_zero.data = y_zero
            intlayernorm.qact.output_zero.data = y_zero

        return y, y_scale, y_zero
    
    def calib_add(
        self,
        qadd: QAdd,
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        pre_scale_1: torch.Tensor,
        pre_scale_2: torch.Tensor,
        pre_zero_1: torch.Tensor | None = None,
        pre_zero_2: torch.Tensor | None = None,
    ):
        y = qadd(x1, x2)

        x1_scale, x1_zero = pre_scale_1, pre_zero_1
        x2_scale, x2_zero = pre_scale_2, pre_zero_2

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=qadd.act_bit,
            quant_mode=qadd.act_quant_mode,
            per_channel=qadd.act_per_channel,
            is_weight=False,
        )

        qadd.x_scale.data = x1_scale
        qadd.y_scale.data = x2_scale
        qadd.o_scale.data = y_scale
        qadd.x_coe.data = x1_scale / y_scale
        qadd.y_coe.data = x2_scale / y_scale

        if qadd.act_quant_mode == "sym":
            pass
        elif qadd.act_quant_mode == "asym":
            qadd.x_zero.data = x1_zero
            qadd.y_zero.data = x2_zero
            qadd.o_zero.data = y_zero

        return y, y_scale, y_zero
    
    def calib_matmul(
        self,
        qmatmul: QMatMul,
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        pre_scale_1: torch.Tensor,
        pre_scale_2: torch.Tensor,
        pre_zero_1: torch.Tensor | None = None,
        pre_zero_2: torch.Tensor | None = None,
    ):
        y = qmatmul(x1, x2)

        x1_scale, x1_zero = pre_scale_1, pre_zero_1
        x2_scale, x2_zero = pre_scale_2, pre_zero_2

        y_scale, y_zero = max_min_quantize_params(
            input_tensor=y,
            bitwidth=qmatmul.act_bit,
            quant_mode=qmatmul.act_quant_mode,
            per_channel=qmatmul.act_per_channel,
            is_weight=False,
        )

        qmatmul.x_scale.data = x1_scale
        qmatmul.y_scale.data = x2_scale
        qmatmul.o_scale.data = y_scale
        # TODO: if per_channel?
        qmatmul.coe.data = x1_scale * x2_scale / y_scale
        if hasattr(qmatmul, "sqrt_dim"):
            qmatmul.coe.data = qmatmul.coe / qmatmul.sqrt_dim

        if qmatmul.act_quant_mode == "sym":
            pass
        elif qmatmul.act_quant_mode == "asym":
            qmatmul.x_zero.data = x1_zero
            qmatmul.y_zero.data = x2_zero
            qmatmul.o_zero.data = y_zero

        return y, y_scale, y_zero

def calibrate():
    pass

calibrator = Calibrator()

def test_calibrate_linear():
    # Symmetric Quantize
    inc = 1024
    outc = 512
    batch = 64
    act_bit = 8
    sample = torch.randn((batch, outc, inc))
    x = torch.randn((outc, inc))
    # TODO: Need Not All Zero Initialization
    linear = nn.Linear(in_features=inc, out_features=outc)

    qlinear = QLinear(in_features=inc, out_features=outc, act_bit=act_bit)
    qlinear.copy_from(linear)

    x_scale, x_zero = max_min_quantize_params(
        input_tensor=sample,
        bitwidth=act_bit,
        quant_mode=qlinear.act_quant_mode,
        per_channel=qlinear.act_per_channel,
        is_weight=False,
    )
    x_int = symmetric_linear_quantize(
        act_bit,
        x,
        x_scale,
        is_weight=False,
    )

    calibrator.calib_linear(
        qlinear, 
        sample, 
        pre_scale=x_scale,
        pre_zero=x_zero
    )

    y = qlinear(x)
    yq = qlinear.forward_int(x_int) * qlinear.output_scale

    torch.testing.assert_close(
        y,
        yq,
        atol=0.05,
        rtol=0.02,
        msg="The Diff of QLinear Calibrate Test:\n" + str(y-yq)
    )

def test_calibrate_conv2d():
    # Symmetric Quantize
    inc = 32
    outc = 64
    batch = 16
    h, w = (64, 64)
    ksz = (3, 3)
    act_bit = 8
    sample = torch.randn((batch, inc, h, w))
    x = torch.randn((1, inc, h, w))
    conv2d = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ksz)
    qconv2d = QConv2d(in_channels=inc, out_channels=outc, kernel_size=ksz, act_bit=act_bit)
    qconv2d.copy_from(conv2d)

    x_scale, x_zero = max_min_quantize_params(
        input_tensor=sample,
        bitwidth=act_bit,
        quant_mode=qconv2d.act_quant_mode,
        per_channel=qconv2d.act_per_channel,
        is_weight=False,
    )
    x_int = symmetric_linear_quantize(
        act_bit,
        x,
        x_scale,
        is_weight=False,
    )

    calibrator.calib_conv2d(
        qconv2d, 
        sample, 
        pre_scale=x_scale,
        pre_zero=x_zero
    )

    y = qconv2d(x)
    yq = qconv2d.forward_int(x_int) * qconv2d.output_scale

    torch.testing.assert_close(
        y,
        yq,
        atol=0.05,
        rtol=0.02,
        msg="The Diff of QConv2d Calibrate Test:\n" + str(y-yq)
    )

def test_calibrate_gelu():
    # Symmetric Quantize
    channel = 1024
    batch = 64
    act_bit = 8
    sample = torch.randn((batch, channel))
    x = torch.randn((1, channel))
    gelu = nn.GELU()
    intgelu = IntGELU(act_bit=act_bit)
    intgelu.copy_from(gelu)

    x_scale, x_zero = max_min_quantize_params(
        input_tensor=sample,
        bitwidth=act_bit,
        quant_mode=intgelu.act_quant_mode,
        per_channel=intgelu.act_per_channel,
        is_weight=False,
    )
    x_int = symmetric_linear_quantize(
        act_bit,
        x,
        x_scale,
        is_weight=False,
    )

    calibrator.calib_gelu(
        intgelu, 
        sample, 
        pre_scale=x_scale,
        pre_zero=x_zero
    )

    y = intgelu(x)
    yq = intgelu.forward_int(x_int) * intgelu.output_scale

    torch.testing.assert_close(
        y,
        yq,
        atol=0.10,
        rtol=0.10,
        msg="The Diff of IntGELU Calibrate Test:\n" + str(y-yq)
    )

def test_calibrate_matmul():
    # Symmetric Quantize
    M = 64
    N = 64
    K = 128
    batch = 64
    act_bit = 8
    sampleA = torch.randn((batch, M, K))
    sampleB = torch.randn((batch, K, N))
    a = torch.randn((1, M, K))
    b = torch.randn((1, K, N))
    matmul = torch.matmul
    qmatmul = QMatMulIsqrtD(dim=K, act_bit=act_bit) # Test / sqrtd to impress the activation
    qmatmul.copy_from(matmul)

    a_scale, a_zero = max_min_quantize_params(
        input_tensor=sampleA,
        bitwidth=act_bit,
        quant_mode=qmatmul.act_quant_mode,
        per_channel=qmatmul.act_per_channel,
        is_weight=False,
    )
    b_scale, b_zero = max_min_quantize_params(
        input_tensor=sampleB,
        bitwidth=act_bit,
        quant_mode=qmatmul.act_quant_mode,
        per_channel=qmatmul.act_per_channel,
        is_weight=False,
    )

    a_int = symmetric_linear_quantize(
        act_bit,
        a,
        a_scale,
        is_weight=False,
    )
    b_int = symmetric_linear_quantize(
        act_bit,
        b,
        b_scale,
        is_weight=False,
    )

    calibrator.calib_matmul(
        qmatmul, 
        sampleA,
        sampleB, 
        pre_scale_1=a_scale,
        pre_scale_2=b_scale,
        pre_zero_1=a_zero,
        pre_zero_2=b_zero,
    )

    y = qmatmul(a, b)
    yq = qmatmul.forward_int(a_int, b_int) * qmatmul.o_scale

    # print(qmatmul.x_scale, qmatmul.y_scale, qmatmul.o_scale)

    # TODO: ??Why
    torch.testing.assert_close(
        y,
        yq,
        atol=0.08,
        rtol=0.10,
        msg="The Diff of QMatMul Calibrate Test:\n" + str(y-yq)
    )

def test_calibrate_softmax():
    # Symmetric Quantize
    channel = 1024
    batch = 64
    act_bit = 8
    sample = torch.randn((batch, channel))
    x = torch.randn((1, channel))
    softmax = nn.Softmax()
    intsoftmax = IntSoftmax(dim=-1, in_act_bit=8, out_act_bit=8)
    intsoftmax.copy_from(softmax)

    x_scale, x_zero = max_min_quantize_params(
        input_tensor=sample,
        bitwidth=act_bit,
        quant_mode=intsoftmax.act_quant_mode,
        per_channel=intsoftmax.act_per_channel,
        is_weight=False,
    )
    x_int = symmetric_linear_quantize(
        act_bit,
        x,
        x_scale,
        is_weight=False,
    )

    calibrator.calib_softmax(
        intsoftmax, 
        sample, 
        pre_scale=x_scale,
        pre_zero=x_zero
    )

    y = intsoftmax(x)
    yq = intsoftmax.forward_int(x_int) * intsoftmax.output_scale

    print(y.sum(-1), yq.sum(-1))

    torch.testing.assert_close(
        y,
        yq,
        atol=0.005,
        rtol=0.05,
        msg="The Diff of IntSoftmax Calibrate Test:\n" + str(y-yq)
    )

def test_calibrate_layernorm():
# Symmetric Quantize
    channel = 192
    len = 256
    batch = 64
    act_bit = 8
    sample = torch.randn((batch, len, channel))
    x = torch.randn((1, len, channel))
    layernorm = nn.LayerNorm(normalized_shape=(channel,))
    intlayernorm = IntLayerNorm(normalized_shape=(channel,), in_act_bit=act_bit, out_act_bit=act_bit)
    intlayernorm.copy_from(layernorm)

    x_scale, x_zero = max_min_quantize_params(
        input_tensor=sample,
        bitwidth=act_bit,
        quant_mode=intlayernorm.act_quant_mode,
        per_channel=intlayernorm.act_per_channel,
        is_weight=False,
    )
    x_int = symmetric_linear_quantize(
        act_bit,
        x,
        x_scale,
        is_weight=False,
    )

    calibrator.calib_layernorm(
        intlayernorm, 
        sample, 
        pre_scale=x_scale,
        pre_zero=x_zero
    )

    y = intlayernorm(x)
    yq = intlayernorm.forward_int(x_int) * intlayernorm.output_scale

    torch.testing.assert_close(
        y,
        yq,
        atol=0.08,
        rtol=0.10,
        msg="The Diff of IntLayerNorm Calibrate Test:\n" + str(y-yq)
    )

def test_calibrate_add():
    # Symmetric Quantize
    len = 64
    dim = 128
    batch = 64
    act_bit = 8
    sampleA = torch.randn((batch, len, dim))
    sampleB = torch.randn((batch, len, dim))
    a = torch.randn((1, len, dim))
    b = torch.randn((1, len, dim))
    add = torch.add
    qadd = QAdd(act_bit=act_bit)
    qadd.copy_from(add)

    a_scale, a_zero = max_min_quantize_params(
        input_tensor=sampleA,
        bitwidth=act_bit,
        quant_mode=qadd.act_quant_mode,
        per_channel=qadd.act_per_channel,
        is_weight=False,
    )
    b_scale, b_zero = max_min_quantize_params(
        input_tensor=sampleB,
        bitwidth=act_bit,
        quant_mode=qadd.act_quant_mode,
        per_channel=qadd.act_per_channel,
        is_weight=False,
    )

    a_int = symmetric_linear_quantize(
        act_bit,
        a,
        a_scale,
        is_weight=False,
    )
    b_int = symmetric_linear_quantize(
        act_bit,
        b,
        b_scale,
        is_weight=False,
    )

    calibrator.calib_add(
        qadd, 
        sampleA,
        sampleB, 
        pre_scale_1=a_scale,
        pre_scale_2=b_scale,
        pre_zero_1=a_zero,
        pre_zero_2=b_zero,
    )

    y = qadd(a, b)
    yq = qadd.forward_int(a_int, b_int) * qadd.o_scale

    torch.testing.assert_close(
        y,
        yq,
        atol=0.08,
        rtol=0.10,
        msg="The Diff of QMatMul Calibrate Test:\n" + str(y-yq)
    )


if __name__ == "__main__":
    # test_calibrate_linear()
    # test_calibrate_conv2d()
    # test_calibrate_gelu()
    # test_calibrate_matmul()
    # test_calibrate_softmax()
    # test_calibrate_layernorm()
    # test_calibrate_add()
    pass