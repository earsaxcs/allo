# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods

import operator
import inspect
import math

try:
    import torch
    from torch import fx
    from torch.nn import functional as F
    from torch.fx.graph_module import GraphModule
    from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata
    from .tracer import AlloTracer
except ImportError:
    pass
from .library import (
    CoreAttention_lib, 
    KVCache_lib, 
    ViTGetFirstToken_float32_lib, 
    ViTGetFirstToken_int8_lib,
    ViTTokenExpand_float32_lib,
    ViTTokenExpand_int8_lib
)
from ..quant.quant_modules import *
from .. import dsl
from ..ir import types
from ..customize import customize
from ..quant.quant_config import DEFAULT_ACT_BIT, DEFAULT_WEIGHT_BIT, DEFAULT_BIAS_BIT
from ..ops.vit import ViTGetFirstToken, ViTTokenExpand


# maybe used in pytorch_vivado.py
def _process_quantized_params(gm, global_vars, quant_config=None):
    """
    处理量化模块的参数和buffers，将浮点权重/偏置替换为整数版本，
    并注入所有量化相关的scale/zero常量到global_vars中。
    
    Scale参数会被转换为定点表示：scale = sign * coe * 2^(-rshift)
    其中 coe ∈ [0.5, 1)，存储为定点数（使用 int32/int64）
    
    Args:
        gm: GraphModule，包含traced的模型
        global_vars: 全局变量字典，将被就地修改
    
    Returns:
        None (就地修改global_vars)
    """
    import numpy as np
    if quant_config is None:
        raise ValueError("quant_config must be provided to process quantized parameters")
    
    def float_to_fixed_point(scale_float, fixed_bits=16):
        """
        将浮点scale转换为定点表示：scale = sign * coe * 2^(-rshift)
        
        当 scale < 1: rshift > 0（右移）
        当 scale > 1: rshift < 0（左移，硬件上实现为左移 |rshift|）
        
        Args:
            scale_float: 浮点scale值或numpy数组
            fixed_bits: 定点系数的位宽，默认16位
        
        Returns:
            sign: 符号位 (±1)，int8
            coe_fixed: 定点系数，int16/32/64，范围 [2^(fixed_bits-1), 2^fixed_bits)，对应浮点 [0.5, 1)
            rshift: 移位数，int16，范围 [-32767, 32767]
                    > 0: 右移（scale < 1）
                    < 0: 左移（scale > 1）
                    = 0: 不移位（scale ≈ 1）
        """
        import numpy as np
        
        dtype = np.int32 if fixed_bits < 32 else np.int64
        scale_float = np.asarray(scale_float, dtype=np.float64)
        
        # 处理零值
        eps = 1e-20
        scale_float = np.where(np.abs(scale_float) < eps, eps, scale_float)
        
        # 提取符号
        sign = np.sign(scale_float).astype(dtype)
        abs_scale = np.abs(scale_float)
        
        # ========== 核心算法 ==========
        # 目标：将 abs_scale 归一化到 [0.5, 1) 范围
        # abs_scale * 2^(-rshift) = coe ∈ [0.5, 1)
        # 
        # 当 abs_scale < 1:
        #   需要右移：rshift > 0
        #   例如：abs_scale = 0.01 → rshift = 7 → coe = 0.01 * 2^7 = 0.64
        # 
        # 当 abs_scale > 1:
        #   需要左移：rshift < 0
        #   例如：abs_scale = 10 → rshift = -4 → coe = 10 * 2^(-4) = 0.625
    
        log2_abs_scale = np.log2(abs_scale)
        
        # rshift 使得 log2(coe) ∈ [-1, 0)
        # log2(abs_scale) - rshift ∈ [-1, 0)
        # rshift ∈ (log2(abs_scale), log2(abs_scale) + 1]
        rshift = np.ceil(-log2_abs_scale).astype(dtype)
        
        # ========== 不再强制 rshift >= 0！允许负数 ==========
        # 对于 abs_scale > 1：
        #   log2(abs_scale) > 0
        #   rshift = ceil(-log2(abs_scale)) < 0
    
        # 计算归一化系数 coe ∈ [0.5, 1)
        coe_float = abs_scale * (2.0 ** rshift)
        
        # 处理浮点精度导致的边界问题
        # 理论上 coe_float ∈ [0.5, 1)，但可能因为精度问题略微超出
        iteration_count = 0
        max_iterations = 64  # 防止无限循环
        
        # 注意：这里的逻辑是 coe >= 1.0 时 rshift 减1（相当于除以2），coe < 0.5 时 rshift 加1（相当于乘以2）
        # 因为 scale = coe * 2^(-rshift)，所以：
        #   - rshift 减小 → 2^(-rshift) 增大 → scale 增大 → coe 需要减小（除以2）
        #   - rshift 增大 → 2^(-rshift) 减小 → scale 缩小 → coe 需要增大（乘以2）
        while np.any(coe_float >= 1.0) and iteration_count < max_iterations:
            mask = coe_float >= 1.0
            rshift = np.where(mask, rshift - 1, rshift)
            coe_float = np.where(mask, coe_float / 2.0, coe_float)
            iteration_count += 1
        
        while np.any(coe_float < 0.5) and iteration_count < max_iterations:
            mask = coe_float < 0.5
            rshift = np.where(mask, rshift + 1, rshift)
            coe_float = np.where(mask, coe_float * 2.0, coe_float)
            iteration_count += 1
        
        if iteration_count >= max_iterations:
            print(f"Warning: Float normalization failed to converge")
        
        # coe_float ∈ [0.5, 1) → coe_fixed ∈ [2^16, 2^17)
        # 注意：使用 uint 表示更合理，但为了兼容性暂时用 int 返回
        max_fixed_value = (1 << fixed_bits) - 1  # 2^17 - 1 = 131071
        min_fixed_value = (1 << (fixed_bits - 1))  # 2^16 = 65536
        
        coe_fixed = np.round(coe_float * (1 << fixed_bits)).astype(dtype)
        coe_fixed = np.clip(coe_fixed, min_fixed_value, max_fixed_value)
        
        # ========== 检查 rshift 范围（int16: -32768 ~ 32767）==========
        max_rshift = 32767
        min_rshift = -32768
        
        if np.any(rshift > max_rshift) or np.any(rshift < min_rshift):
            overflow_indices = np.where((rshift > max_rshift) | (rshift < min_rshift))[0]
            overflow_values = scale_float.flat[overflow_indices] if scale_float.ndim > 0 else [scale_float]
            overflow_rshifts = rshift.flat[overflow_indices] if rshift.ndim > 0 else [rshift]
            
            print(f"ERROR: rshift overflow detected!")
            print(f"  rshift range: [{np.min(rshift)}, {np.max(rshift)}]")
            print(f"  int16 range: [{min_rshift}, {max_rshift}]")
            print(f"  Problematic values (first 5):")
            for i, (val, rs) in enumerate(zip(overflow_values[:5], overflow_rshifts[:5])):
                print(f"    scale={val:.2e}, rshift={rs}")
            print(f"  This indicates extreme scale values. Consider re-calibration.")
            
            # 钳制到有效范围
            rshift = np.clip(rshift, min_rshift, max_rshift)
        
        # ========== 验证精度 ==========
        reconstructed_scale = sign * (coe_fixed / (1 << fixed_bits)) * (2.0 ** (-rshift.astype(np.float64)))
        relative_error = np.abs((reconstructed_scale - scale_float) / (scale_float + eps))
        max_error = np.max(relative_error)
        
        # 在验证精度部分增强输出
        if max_error > 0.01:
            error_indices = np.where(relative_error > 0.01)[0]
            print(f"WARNING: Fixed-point conversion max relative error: {max_error:.2%}")
            print(f"  Number of problematic values: {len(error_indices)}")
            print(f"  Original scales (first 5): {scale_float.flat[error_indices[:5]]}")
            print(f"  Reconstructed (first 5): {reconstructed_scale.flat[error_indices[:5]]}")
            print(f"  Relative errors (first 5): {relative_error.flat[error_indices[:5]]}")
            print(f"  rshift values (first 5): {rshift.flat[error_indices[:5]]}")
            print(f"  coe_fixed values (first 5): {coe_fixed.flat[error_indices[:5]]}")
        
        # 此处均以i32/i64返回
        # 由外部负责转换到需要的类型

        # NOTE: 合并coe_fixed的fixed_bits到rshift里，之后coe_fixed当成一个纯粹的介于[2^(fixed_bits-1), 2^fixed_bits)的整数看待
        rshift = rshift + fixed_bits
        return sign, coe_fixed, rshift
    
    # 定义需要处理的量化模块类型
    quantized_module_types = (
        QLinear, 
        QConv2d, 
        IntGELU, 
        IntSoftmax, 
        IntLayerNorm, 
        QAdd, 
        QMatMul, 
        QMatMulIsqrtD, 
        # 可以继续添加其他量化模块
    )
    
    # 遍历所有命名模块
    for module_name, module in gm.named_modules():
        if not isinstance(module, quantized_module_types):
            continue
            
        # 模块名转换为变量名格式
        var_prefix = "g_" + module_name.replace(".", "_")
        
        # ========== 处理权重 (weight) ==========
        if hasattr(module, 'weight') and hasattr(module, 'weight_int'):
            if hasattr(module.weight_int, 'data'):
                weight_int_data = module.weight_int.data.detach().numpy()
                # 简单检查：如果weight_int全为0，可能未校准
                if not np.any(weight_int_data):
                    print(f"Warning: {module_name}.weight_int appears to be all zeros. "
                          f"Did you run Calibrator.calibrate()?")
                
                # 替换weight为weight_int（整数版本）
                weight_key = var_prefix + "_weight_int"
                global_vars[weight_key] = weight_int_data.astype(np.int8)
                
                # 注入weight_scale（定点表示）
                if hasattr(module, 'weight_scale'):
                    scale_data = module.weight_scale.data.detach().numpy()
                    sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
                    
                    global_vars[var_prefix + "_weight_scale_sign"] = sign.astype(np.int8)
                    global_vars[var_prefix + "_weight_scale_coe"] = coe.astype(np.int32)
                    global_vars[var_prefix + "_weight_scale_rshift"] = rshift.astype(np.int16)
        
        # ========== 处理偏置 (bias) ==========
        if hasattr(module, 'bias') and module.bias is not None:
            dtype_str = "int32"
            if hasattr(module, 'bias_int'):
                bias_int = module.bias_int.data
                if isinstance(module, QLinear):
                    if module.wgt_per_channel:
                        raise NotImplementedError("wgt_per_channel not supported")
                    else:
                        # NOTE: Here is hardcoded for classifier, please note if you change the model's module name, you need to change this line
                        if 'classifier' not in module_name:
                            bias_int = bias_int[None, :].expand(quant_config.seq_len, -1).contiguous() # Also need to broadcast when build
                        else:
                            bias_int = bias_int[None, :].contiguous()
                        dtype_str = f"int{module.bias_bit}"
                    # NOTE: previously this path was effectively unconditional due to `act_per_token or not act_per_token`.
                bias_int_data = bias_int.detach().numpy()
                bias_key = var_prefix + "_bias_int"
                global_vars[bias_key] = bias_int_data.astype(dtype_str)
                
                # 注入bias_scale（定点表示）
                if hasattr(module, 'bias_scale'):
                    scale_data = module.bias_scale.data.detach().numpy()
                    sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
                    
                    global_vars[var_prefix + "_bias_scale_sign"] = sign.astype(np.int8)
                    global_vars[var_prefix + "_bias_scale_coe"] = coe.astype(np.int32)
                    global_vars[var_prefix + "_bias_scale_rshift"] = rshift.astype(np.int16)
        
        # ========== 处理激活量化相关的scale（定点表示）==========
        
        # input_scale
        if hasattr(module, 'input_scale'):
            scale_data = module.input_scale.data.detach().numpy()
            sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
            
            global_vars[var_prefix + "_input_scale_sign"] = sign.astype(np.int8)
            global_vars[var_prefix + "_input_scale_coe"] = coe.astype(np.int32)
            global_vars[var_prefix + "_input_scale_rshift"] = rshift.astype(np.int16)
        
        # output_scale
        if hasattr(module, 'output_scale'):
            scale_data = module.output_scale.data.detach().numpy()
            sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
            
            global_vars[var_prefix + "_output_scale_sign"] = sign.astype(np.int8)
            global_vars[var_prefix + "_output_scale_coe"] = coe.astype(np.int32)
            global_vars[var_prefix + "_output_scale_rshift"] = rshift.astype(np.int16)
            
            # Compute output_scale inverse: 1/output_scale
            # Use epsilon to avoid division by zero (guard only near-zero values)
            eps = 1e-10
            scale_inv_data = np.where(np.abs(scale_data) < eps, 0.0, 1.0 / scale_data)
            sign_inv, coe_inv, rshift_inv = float_to_fixed_point(scale_inv_data, fixed_bits=quant_config.scale_fixed_bits)
            
            global_vars[var_prefix + "_output_scale_inv_sign"] = sign_inv.astype(np.int8)
            global_vars[var_prefix + "_output_scale_inv_coe"] = coe_inv.astype(np.int32)
            global_vars[var_prefix + "_output_scale_inv_rshift"] = rshift_inv.astype(np.int16)
        
        # fused_scale（最关键：用于运行时缩放）
        if hasattr(module, 'fused_scale'):
            scale_data = module.fused_scale.data.detach().numpy()
            sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
            
            global_vars[var_prefix + "_fused_scale_sign"] = sign.astype(np.int8)
            global_vars[var_prefix + "_fused_scale_coe"] = coe.astype(np.int32)
            global_vars[var_prefix + "_fused_scale_rshift"] = rshift.astype(np.int16)
        
        # input_zero / output_zero（非对称量化，保持整数）
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            if hasattr(module.input_zero, 'data'):
                zero_data = module.input_zero.data.detach().numpy()
                global_vars[var_prefix + "_input_zero"] = zero_data.astype(np.int8)
        
        if hasattr(module, 'output_zero') and module.output_zero is not None:
            if hasattr(module.output_zero, 'data'):
                zero_data = module.output_zero.data.detach().numpy()
                global_vars[var_prefix + "_output_zero"] = zero_data.astype(np.int8)
        
        # ========== 特殊处理：IntLayerNorm ==========
        if isinstance(module, IntLayerNorm):
            if hasattr(module, 'layernorm_scale'):
                scale_data = module.layernorm_scale.data.detach().numpy()
                sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
                
                global_vars[var_prefix + "_layernorm_scale_sign"] = sign.astype(np.int8)
                global_vars[var_prefix + "_layernorm_scale_coe"] = coe.astype(np.int32)
                global_vars[var_prefix + "_layernorm_scale_rshift"] = rshift.astype(np.int16)
            
            # it will be publicly assign above, so no need to assign again
            # if hasattr(module, 'bias_int'):
            #     bias_int_data = module.bias_int.data.detach().numpy()
            #     global_vars[var_prefix + "_bias_int"] = bias_int_data.astype(np.int32)
        
        # ========== 特殊处理：QAdd/QMatMul的双输入scale ==========
        if isinstance(module, (QAdd, QMatMul, QMatMulIsqrtD)):
            # x_scale, y_scale, fused_scale, o_scale
            # NOTE: fused_scale is semantically x_scale * y_scale / o_scale.
            # For QMatMulIsqrtD, fused_scale is expected to have already fused 1/sqrt(d)
            # inside the quant module calibration logic.
            # For QAdd, fused_scale is Not used
            for scale_name in ['x_scale', 'y_scale', 'fused_scale', 'o_scale']:
                if hasattr(module, scale_name):
                    scale_data = getattr(module, scale_name).data.detach().numpy()
                    sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
                    
                    global_vars[var_prefix + f"_{scale_name}_sign"] = sign.astype(np.int8)
                    global_vars[var_prefix + f"_{scale_name}_coe"] = coe.astype(np.int32)
                    global_vars[var_prefix + f"_{scale_name}_rshift"] = rshift.astype(np.int16)
                    
                    # Compute inverse for o_scale
                    if scale_name == 'o_scale':
                        eps = 1e-10
                        # If this is the ISqrtD variant, we must fuse the 1/sqrt(dim)
                        # factor into the scaling (otherwise emitting plain qmatmul
                        # would miss the normalization).
                        # if isinstance(module, QMatMulIsqrtD) and hasattr(module, 'sqrt_dim'):
                        #     sqrt_dim = module.sqrt_dim.detach().cpu().numpy()
                        #     scale_inv_data = 1.0 / (scale_data * sqrt_dim + eps)
                        # else:
                        #     scale_inv_data = 1.0 / (scale_data + eps)
                        
                        # NOTE: Currently just directly records output_scale_inv
                        # NOTE: But it's true that we can't distinguish them after here in MLIR, it's a real risk so you need to focus on this when you use other feature out of fused_scale
                        scale_inv_data = 1.0 / (scale_data + eps)
                        sign_inv, coe_inv, rshift_inv = float_to_fixed_point(scale_inv_data, fixed_bits=quant_config.scale_fixed_bits)
                        
                        global_vars[var_prefix + "_o_scale_inv_sign"] = sign_inv.astype(np.int8)
                        global_vars[var_prefix + "_o_scale_inv_coe"] = coe_inv.astype(np.int32)
                        global_vars[var_prefix + "_o_scale_inv_rshift"] = rshift_inv.astype(np.int16)
            
            # zero points（保持整数）
            for zero_name in ['x_zero', 'y_zero', 'o_zero']:
                if hasattr(module, zero_name):
                    zero_attr = getattr(module, zero_name)
                    if zero_attr is not None and hasattr(zero_attr, 'data'):
                        zero_data = zero_attr.data.detach().numpy()
                        global_vars[var_prefix + f"_{zero_name}"] = zero_data.astype(np.int8)
        
        # ========== 特殊处理：IntSoftmax ==========
        if isinstance(module, IntSoftmax):
            if hasattr(module, 'softmax_scale'):
                scale_data = module.softmax_scale.data.detach().numpy()
                sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
                
                global_vars[var_prefix + "_softmax_scale_sign"] = sign.astype(np.int8)
                global_vars[var_prefix + "_softmax_scale_coe"] = coe.astype(np.int32)
                global_vars[var_prefix + "_softmax_scale_rshift"] = rshift.astype(np.int16)
        
        # ========== 特殊处理：IntGELU ==========
        if isinstance(module, IntGELU):
            # IntGELU 需要 4 组 scale: input_scale, gelu_scale, fused_scale, output_scale
            # (input_scale, output_scale 和 fused_scale 已经在通用部分处理)
            if hasattr(module, 'gelu_scale'):
                scale_data = module.gelu_scale.data.detach().numpy()
                sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=quant_config.scale_fixed_bits)
                
                global_vars[var_prefix + "_gelu_scale_sign"] = sign.astype(np.int8)
                global_vars[var_prefix + "_gelu_scale_coe"] = coe.astype(np.int32)
                global_vars[var_prefix + "_gelu_scale_rshift"] = rshift.astype(np.int16)

def _convert_weight_to_int(weight_data, bits=8):
    """
    将伪量化的浮点权重转换为整数类型。
    
    假设输入的浮点权重已经是伪量化完毕的，即数值上是整数但以浮点形式存储。
    例如：tensor([[-3.0, 2.0, 1.0], [0.0, -1.0, 4.0]]) → int8([[-3, 2, 1], [0, -1, 4]])
    
    Args:
        weight_data: numpy 数组，伪量化的浮点权重
        bits: 目标位宽，默认 8
    
    Returns:
        numpy 数组，整数类型的权重
    """
    import numpy as np
    
    # 确定目标类型
    if bits <= 8:
        target_dtype = np.int8
        min_val, max_val = -128, 127
    elif bits <= 16:
        target_dtype = np.int16
        min_val, max_val = -32768, 32767
    else:
        target_dtype = np.int32
        min_val, max_val = -2147483648, 2147483647
    
    # 四舍五入并裁剪到目标范围
    weight_int = np.round(weight_data).astype(np.float64)
    weight_int = np.clip(weight_int, min_val, max_val)
    
    return weight_int.astype(target_dtype)


def from_pytorch(
    model,
    example_inputs,
    leaf_modules=None,
    verbose=False,
    enable_tensor=False,
    target="llvm",
    mode="csim",
    project="top.prj",
    enable_quant=True,  # 量化开关：默认开启
):
    import numpy as np
    
    sig = inspect.signature(model.forward)
    input_names = [
        p.name for i, p in enumerate(sig.parameters.values()) if i < len(example_inputs)
    ]
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    args = []
    args += example_inputs
    for item in concrete_args.values():
        args.append(item)

    tracer = AlloTracer(model, concrete_args=concrete_args, leaf_modules=leaf_modules)
    graph = tracer.trace()
    name = (
        model.__class__.__name__
        if isinstance(model, torch.nn.Module)
        else model.__name__
    )
    gm = GraphModule(tracer.root, graph, name)
    ShapeProp(gm).propagate(*args)
    if verbose:
        print(gm.graph)
    global_vars = {}
    for pymod in (types,):
        global_vars.update({item[0]: item[1] for item in inspect.getmembers(pymod)})
    global_vars.update({"dsl": dsl})
    
    # # 定义量化模块类型
    # quantized_module_types = (
    #     QLinear, QConv2d, IntGELU, IntSoftmax, 
    #     IntLayerNorm, QAdd, QMatMul, QMatMulIsqrtD,
    # )
    
    # # 收集量化模块信息，用于判断参数是否属于量化模块
    # quant_module_prefixes = set()
    # if enable_quant:
    #     for module_name, module in gm.named_modules():
    #         if isinstance(module, quantized_module_types):
    #             quant_module_prefixes.add(module_name)
    
    # # 处理参数 (nn.Parameter)
    # # 对于量化模块的权重，需要将伪量化的浮点转换为整数
    # # 定义需要量化的 embedding 参数后缀
    # # Currently empty for we don't need to quantize embedding or cls token
    # quant_embedding_suffixes = tuple() # ("_embedding", "_cls_token")
    
    # for param_name, param in gm.named_parameters():
    #     new_name = "g_" + param_name.replace(".", "_")
    #     param_data = param.detach().numpy()
        
    #     if enable_quant:
    #         converted = False
    #         # 检查是否是量化模块的权重
    #         for prefix in quant_module_prefixes:
    #             if param_name.startswith(prefix + "."):
    #                 # 这是量化模块的参数
    #                 # 检查是否是权重参数
    #                 param_suffix = param_name[len(prefix)+1:]  # 获取 "weight" 或 "bias" 等
    #                 if param_suffix == "weight":
    #                     # 量化模块的权重，转换为 int8
    #                     param_data = _convert_weight_to_int(param_data, bits=TorchBuilder.QUANT_WEIGHT_BITS)
    #                     converted = True
    #                 elif param_suffix == "bias":
    #                     # 偏置转换为 int32
    #                     param_data = _convert_weight_to_int(param_data, bits=TorchBuilder.QUANT_BIAS_BITS)
    #                     converted = True
    #                 break
            
    #         # 如果不是量化模块参数，检查是否是需要量化的 embedding 参数
    #         if not converted:
    #             for suffix in quant_embedding_suffixes:
    #                 if param_name.endswith(suffix):
    #                     # embedding 参数使用激活值位宽 (int8)
    #                     param_data = _convert_weight_to_int(param_data, bits=TorchBuilder.QUANT_ACT_BITS)
    #                     break
        
    #     global_vars.update({new_name: param_data})
    
    # # 处理 buffers (register_buffer)
    # # 量化模块的 buffer（如 weight_int, bias_int）需要转换为整数类型
    # for buffer_name, buffer in gm.named_buffers():
    #     new_name = "g_" + buffer_name.replace(".", "_")
    #     buffer_data = buffer.detach().numpy()
        
    #     if enable_quant:
    #         # 检查是否是量化模块的 buffer
    #         for prefix in quant_module_prefixes:
    #             if buffer_name.startswith(prefix + "."):
    #                 buffer_suffix = buffer_name[len(prefix)+1:]
    #                 # weight_int 应该转换为 int8
    #                 if buffer_suffix == "weight_int":
    #                     buffer_data = _convert_weight_to_int(buffer_data, bits=TorchBuilder.QUANT_WEIGHT_BITS)
    #                 # bias_int 应该转换为 int32
    #                 elif buffer_suffix == "bias_int":
    #                     buffer_data = _convert_weight_to_int(buffer_data, bits=TorchBuilder.QUANT_BIAS_BITS)
    #                 # scale 和 zero 相关的 buffer 保持原样（会在后续定点转换中处理）
    #                 break
        
    #     global_vars.update({new_name: buffer_data})

    builder = TorchBuilder(gm, example_inputs, leaf_modules, enable_quant=enable_quant)
    code = builder.build()
    s = customize(
        code, verbose=verbose, global_vars=global_vars, enable_tensor=enable_tensor
    )
    mod = s.build(target=target, mode=mode, project=project)
    if verbose:
        print(s.module)
    return mod


def get_var_name(node):
    return node.name if isinstance(node, fx.Node) else node


class TorchBuilder:
    # 量化位宽配置 - 用于确定量化模块的输入/输出/权重类型
    # 可以通过修改这些类属性来调整全局默认位宽
    QUANT_ACT_BITS = DEFAULT_ACT_BIT      # 激活值（输入/输出）的位宽
    QUANT_WEIGHT_BITS = DEFAULT_WEIGHT_BIT   # 权重的位宽
    QUANT_BIAS_BITS = DEFAULT_BIAS_BIT    # 偏置的位宽
    
    def __init__(self, gm, example_inputs, leaf_modules=None, 
                 quant_act_bits=None, quant_weight_bits=None, quant_bias_bits=None,
                 enable_quant=False, quant_config=None):
        self.gm = gm
        self.code = []
        self.input_names = []
        self.input_shapes = []
        self.example_inputs = example_inputs
        self.leaf_modules = leaf_modules
        self.input_args = []
        self.named_params = gm.named_parameters()
        self.named_buffers = gm.named_buffers()
        self.subfunctions = []
        self.output = []
        
        # 量化开关：控制是否启用量化类型转换
        self.enable_quant = enable_quant
        if enable_quant and quant_config is None:
            raise ValueError("QuantConfig is required when enable_quant is True")
        self.quant_config = quant_config
        
        # 实例级别的量化位宽配置（允许覆盖类级别的默认值）
        self.quant_act_bits = quant_act_bits if quant_act_bits is not None else self.QUANT_ACT_BITS
        self.quant_weight_bits = quant_weight_bits if quant_weight_bits is not None else self.QUANT_WEIGHT_BITS
        self.quant_bias_bits = quant_bias_bits if quant_bias_bits is not None else self.QUANT_BIAS_BITS
        
        # 缓存：记录哪些输入节点被量化模块使用
        self._quant_input_nodes = set()
        self._quant_output_nodes = set()
        self._quant_module_prefixes = set()
        
        # 边界检测：记录需要插入 quant/dequant 的位置
        # key: (producer_node_name, consumer_node_name)
        # value: dict with 'type' ('quant'/'dequant'), 'scale_source' (量化模块名用于获取scale)
        self._boundary_conversions = {}
        # 映射：节点名 -> 该节点对应的量化模块信息（用于获取scale）
        self._node_quant_info = {}
        # 映射：节点名 -> 输出类型 ('int8' or 'float32')
        self._node_output_types = {}
        
        # 只有启用量化时才进行预处理
        if self.enable_quant:
            self._preprocess_quantized_io()
            self._detect_quant_boundaries()
    
    def _preprocess_quantized_io(self):
        """
        预处理：分析计算图，找出哪些<<<placeholder>>>节点被量化模块使用，
        以及哪些output节点是量化模块的输出。
        这样在生成函数签名和返回类型时，可以正确地使用int8类型。
        """
        # 定义量化模块类型
        quantized_module_types = (
            QLinear, 
            QConv2d, 
            IntGELU, 
            IntSoftmax, 
            IntLayerNorm, 
            QAdd, 
            QMatMul, 
            QMatMulIsqrtD, 
        )
        
        # 收集量化模块前缀
        for module_name, module in self.gm.named_modules():
            if isinstance(module, quantized_module_types):
                self._quant_module_prefixes.add(module_name)
        
        # 遍历所有节点
        for node in self.gm.graph.nodes:
            if node.op == 'call_module':
                module = dict(self.gm.named_modules()).get(node.target)
                if module is not None and isinstance(module, quantized_module_types):
                    # 记录该量化模块的输入节点
                    for arg in node.args:
                        if isinstance(arg, fx.Node):
                            # 追溯到placeholder节点
                            self._trace_to_placeholder(arg, self._quant_input_nodes)
                    
                    # 记录该量化模块的输出节点名
                    self._quant_output_nodes.add(node.name)
                    # 记录节点的量化信息
                    self._node_quant_info[node.name] = {
                        'module_name': node.target,
                        'module': module,
                        'is_quant': True
                    }
    
    def _detect_quant_boundaries(self):
        """
        检测量化/非量化边界，识别需要插入 quant/dequant 的位置。
        
        边界情况：
        1. 非量化节点的输出 -> 量化模块的输入：需要插入 quant
        2. 量化模块的输出 -> 非量化节点的输入：需要插入 dequant
        3. call_function, call_method (如 cat, transpose) 与量化模块之间的边界
        
        使用边界处量化模块的 input_scale (quant) 或 output_scale (dequant)
        """
        quantized_module_types = (
            QLinear, QConv2d, IntGELU, IntSoftmax, 
            IntLayerNorm, QAdd, QMatMul, QMatMulIsqrtD,
        )

        leaf_module_function_types = (ViTGetFirstToken, ViTTokenExpand)

        supported_tensor_ops = ("cat", "transpose", "view", "reshape", "permute", "flatten")
        
        modules_dict = dict(self.gm.named_modules())
        
        # 首先建立一个映射：节点 -> 输出类型（int8 或 float32）
        # 这样可以追溯 call_function, call_method 节点 的输出类型
        node_output_types = {}  # node_name -> 'int8' or 'float32'
        
        # 第一遍：标记所有量化模块的输出为 int8
        for node in self.gm.graph.nodes:
            if node.op == 'call_module':
                module = modules_dict.get(node.target)
                if module is not None and isinstance(module, quantized_module_types):
                    node_output_types[node.name] = f'int{self.quant_act_bits}'
                else:
                    node_output_types[node.name] = 'float32'
            elif node.op == 'placeholder':
                # placeholder 默认是 float32（除非在 _quant_input_nodes 中）
                node_output_types[node.name] = 'float32'
        
        # 第二遍：推断 call_function, call_method 的输出类型
        # call_function 通常保持输入类型（如 cat, transpose, view 等）
        for node in self.gm.graph.nodes:
            if node.op == 'call_function' or \
            node.op == 'call_method' or \
            (node.op == 'call_module' and isinstance(modules_dict.get(node.target, None), leaf_module_function_types)):
                # 检查第一个输入参数的类型
                # 对于大多数操作（cat, transpose, view 等），输出类型 = 输入类型
                input_type = 'float32'  # 默认
                for arg in node.args:
                    if isinstance(arg, fx.Node) and arg.name in node_output_types:
                        input_type = node_output_types[arg.name]
                        break  # 使用第一个有效输入的类型
                node_output_types[node.name] = input_type
        
        # 第三遍：检测边界并插入转换
        # TODO: 边界检测还没修改
        for node in self.gm.graph.nodes:
            if node.op == 'call_module' and not isinstance(modules_dict.get(node.target, None), leaf_module_function_types):
                module = modules_dict.get(node.target)
                if module is None:
                    continue
                
                is_quant_module = isinstance(module, quantized_module_types)
                
                if is_quant_module:
                    # 检查输入：非 int8 -> 量化模块 = 需要 quant
                    for i, arg in enumerate(node.args):
                        if not isinstance(arg, fx.Node):
                            continue
                        
                        # 获取输入的输出类型
                        input_type = node_output_types.get(arg.name, 'float32')
                        
                        if input_type == 'float32':
                            # float32 -> 量化模块：需要 quant
                            key = (arg.name, node.name, i)
                            self._boundary_conversions[key] = {
                                'type': 'quant',
                                'consumer_module': node.target,
                                'consumer_module_obj': module,
                                'input_index': i
                            }
                else:
                    # 非量化模块：检查输入是否来自 int 源
                    for i, arg in enumerate(node.args):
                        if not isinstance(arg, fx.Node):
                            continue
                        
                        # 获取输入的输出类型
                        input_type = node_output_types.get(arg.name, 'float32')
                        
                        if input_type == f'int{self.quant_act_bits}':
                            # int8 -> 非量化模块：需要 dequant
                            # 需要找到产生 int8 的量化模块
                            producer_module_name, producer_module_obj = self._find_producer_quant_module(arg, modules_dict, quantized_module_types)
                            
                            if producer_module_name:
                                key = (arg.name, node.name, i)
                                self._boundary_conversions[key] = {
                                    'type': 'dequant',
                                    'producer_module': producer_module_name,
                                    'producer_module_obj': producer_module_obj,
                                    'input_index': i
                                }
            
            elif node.op == 'call_function' or \
            node.op == 'call_method' or \
            (node.op == 'call_module' and isinstance(modules_dict.get(node.target, None), leaf_module_function_types)):
                if node.op in ('call_function', 'call_method') and node.name.split('_')[0] not in supported_tensor_ops:
                    continue
                # call_function, call_method 也需要检查其输入
                for i, arg in enumerate(node.args):
                    if not isinstance(arg, fx.Node):
                        continue
                    
                    # 获取输入的输出类型
                    input_type = node_output_types.get(arg.name, 'float32')
                    
                    # 上面几个列出的 call_function, call_method 通常期望 输入等于输出类型
                    if input_type != f'int{self.quant_act_bits}':
                        # 需要找到产生 int8 的量化模块
                        producer_module_name, producer_module_obj = self._find_producer_quant_module(arg, modules_dict, quantized_module_types)
                        
                        if producer_module_name:
                            key = (arg.name, node.name, i)
                            self._boundary_conversions[key] = {
                                'type': 'dequant',
                                'producer_module': producer_module_name,
                                'producer_module_obj': producer_module_obj,
                                'input_index': i
                            }

        # 保存到实例变量，供后续 build 方法查询
        self._node_output_types = node_output_types
    
    def _find_producer_quant_module(self, node, modules_dict, quantized_module_types):
        """
        追溯节点，找到产生该节点输出的量化模块
        
        Args:
            node: 当前节点
            modules_dict: 模块字典
            quantized_module_types: 量化模块类型元组
        
        Returns:
            (module_name, module_obj): 量化模块的名称和对象，如果未找到则返回 (None, None)
        """
        # 如果节点本身就是量化模块，直接返回
        if node.op == 'call_module':
            module = modules_dict.get(node.target)
            if module is not None and isinstance(module, quantized_module_types):
                return node.target, module
        
        # 如果是 call_function 或 call_method，追溯其输入
        if node.op == 'call_function' or node.op == 'call_method':
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    result = self._find_producer_quant_module(arg, modules_dict, quantized_module_types)
                    if result[0] is not None:
                        return result
        
        return None, None
    
    def _get_quant_mode_from_module(self, module):
        """从量化模块获取量化模式编码"""
        # 默认：symmetric, per-tensor
        mode = 0
        if hasattr(module, 'quant_mode'):
            qmode = module.quant_mode
            if qmode == 'asym':
                mode |= 1  # asymmetric
            if hasattr(module, 'per_token') and module.per_token:
                mode |= 2  # per-token
        return mode
    
    def _generate_boundary_conversion(self, conversion_info, producer_name, consumer_name, input_idx):
        """生成边界转换的代码（quant 或 dequant）
        
        Returns:
            (new_var_name, code_line) 或 None 如果不需要转换
        """
        conv_type = conversion_info['type']
        
        if conv_type == 'quant':
            # 非量化 -> 量化：使用消费者（量化模块）的 input_scale
            module_name = conversion_info['consumer_module'].replace(".", "_")
            module_obj = conversion_info['consumer_module_obj']
            quant_mode = self._get_quant_mode_from_module(module_obj)
            
            # 生成新变量名
            new_var_name = f"{producer_name}_quant_{consumer_name}_{input_idx}"
            
            # 根据模块类型和输入索引选择正确的 scale 名称
            scale_base_name = self._get_input_scale_name(module_obj, input_idx)
            scale_sign = f"{module_name}_{scale_base_name}_sign"
            scale_coe = f"{module_name}_{scale_base_name}_coe"
            scale_rshift = f"{module_name}_{scale_base_name}_rshift"
            
            # 检查是否需要 zero point
            zero = None
            zero_attr_name = self._get_input_zero_name(module_obj, input_idx)
            if zero_attr_name and hasattr(module_obj, zero_attr_name):
                zero_val = getattr(module_obj, zero_attr_name)
                if zero_val is not None:
                    zero = f"{module_name}_{zero_attr_name}"
            
            code = self.generate_quant_call(
                new_var_name, producer_name, quant_mode,
                scale_sign, scale_coe, scale_rshift, zero
            )
            return new_var_name, code
        
        elif conv_type == 'dequant':
            # 量化 -> 非量化：使用生产者（量化模块）的 output_scale
            module_name = conversion_info['producer_module'].replace(".", "_")
            module_obj = conversion_info['producer_module_obj']
            quant_mode = self._get_quant_mode_from_module(module_obj)
            
            # 生成新变量名
            new_var_name = f"{producer_name}_dequant_{consumer_name}_{input_idx}"
            
            # 根据模块类型选择正确的 output scale 名称
            scale_base_name = self._get_output_scale_name(module_obj)
            scale_sign = f"{module_name}_{scale_base_name}_sign"
            scale_coe = f"{module_name}_{scale_base_name}_coe"
            scale_rshift = f"{module_name}_{scale_base_name}_rshift"
            
            # 检查是否需要 zero point
            zero = None
            zero_attr_name = self._get_output_zero_name(module_obj)
            if zero_attr_name and hasattr(module_obj, zero_attr_name):
                zero_val = getattr(module_obj, zero_attr_name)
                if zero_val is not None:
                    zero = f"{module_name}_{zero_attr_name}"
            
            code = self.generate_dequant_call(
                new_var_name, producer_name, quant_mode,
                scale_sign, scale_coe, scale_rshift, zero
            )
            return new_var_name, code
        
    # NOTE: You need to determine the correct input scale name based on module type and input index. Update this method if the corresponding module is updated.
    def _get_input_scale_name(self, module, input_idx):
        """根据模块类型和输入索引获取正确的输入 scale 名称"""
        # 双输入操作：QAdd, QMatMul, QMatMulIsqrtD
        if isinstance(module, (QAdd, QMatMul, QMatMulIsqrtD)):
            if input_idx == 0:
                return "x_scale"
            elif input_idx == 1:
                return "y_scale"
            else:
                # 默认使用 x_scale（不应该到这里）
                return "x_scale"
        
        # 单输入操作：QLinear, QConv2d, IntGELU, IntSoftmax, IntLayerNorm
        return "input_scale"
    
    def _get_input_zero_name(self, module, input_idx):
        """根据模块类型和输入索引获取正确的输入 zero point 名称"""
        # 双输入操作：QAdd, QMatMul, QMatMulIsqrtD
        if isinstance(module, (QAdd, QMatMul, QMatMulIsqrtD)):
            if input_idx == 0:
                return "x_zero"
            elif input_idx == 1:
                return "y_zero"
            else:
                return "x_zero"
        
        # 单输入操作
        return "input_zero"
    
    def _get_output_scale_name(self, module):
        """根据模块类型获取正确的输出 scale 名称"""
        # 双输入操作：QAdd, QMatMul, QMatMulIsqrtD 使用 o_scale
        if isinstance(module, (QAdd, QMatMul, QMatMulIsqrtD)):
            return "o_scale"
        
        # 其他模块使用 output_scale
        return "output_scale"
    
    def _get_output_zero_name(self, module):
        """根据模块类型获取正确的输出 zero point 名称"""
        # 双输入操作：QAdd, QMatMul, QMatMulIsqrtD 使用 o_zero
        if isinstance(module, (QAdd, QMatMul, QMatMulIsqrtD)):
            return "o_zero"
        
        # 其他模块使用 output_zero
        return "output_zero"
    def _trace_to_placeholder(self, node, result_set):
        """
        追溯节点到placeholder，将路径上的节点都标记为量化相关。
        """
        if node.op == 'placeholder':
            result_set.add(node.name)
        elif node.op == 'call_module':
            # 检查是否是量化模块的输出
            module = dict(self.gm.named_modules()).get(node.target)
            quantized_module_types = (
                QLinear, QConv2d, IntGELU, IntSoftmax, 
                IntLayerNorm, QAdd, QMatMul, QMatMulIsqrtD,
            )
            if module is not None and isinstance(module, quantized_module_types):
                # 量化模块的输出已经是int8，不需要再追溯
                return
        # 继续追溯输入
        for arg in node.args:
            if isinstance(arg, fx.Node):
                self._trace_to_placeholder(arg, result_set)
    
    def _get_quant_dtype_str(self, bits, signed=True):
        """
        根据位宽返回对应的类型字符串。
        """
        if signed:
            return f"int{bits}"
        else:
            return f"uint{bits}"
    
    # Deprecated Currently not affected
    def _is_quantized_input(self, input_name):
        """
        检查某个输入是否被量化模块使用（仅在启用量化时有效）。
        """
        return False and self.enable_quant and input_name in self._quant_input_nodes
    
    # Deprecated Currently not affected
    def _is_quantized_output(self, output_name):
        """
        检查某个输出是否来自量化模块（仅在启用量化时有效）。
        """
        return False and self.enable_quant and output_name in self._quant_output_nodes
    
    def _is_quant_module_param(self, param_name):
        """
        检查某个参数是否属于量化模块（仅在启用量化时有效）。
        返回 (is_quant, param_suffix) 或 (False, None)
        """
        if not self.enable_quant:
            return False, None
        for prefix in self._quant_module_prefixes:
            if param_name.startswith(prefix + "."):
                param_suffix = param_name[len(prefix)+1:]
                return True, param_suffix
        return False, None

    # Currently not affected
    def _is_quant_embedding_param(self, param_name):
        """
        检查某个参数是否是需要量化的 embedding 参数。
        这些参数（如 cls_token、）虽然不属于量化模块，
        但会被传递给量化流程（如 ViTTokenExpand -> QAdd），因此可能需要量化。
        仅在启用量化时有效。
        """
        if not self.enable_quant:
            return False
        # 定义需要量化的 embedding 参数后缀
        quant_embedding_suffixes = tuple()
        # 检查参数名是否以这些后缀结尾
        for suffix in quant_embedding_suffixes:
            if param_name.endswith(suffix):
                return True
        return False

    def build(self):
        for node in self.gm.graph.nodes:
            self(node)
        for i, x in enumerate(self.example_inputs):
            if isinstance(x, torch.Tensor):
                self.input_shapes.append(x.shape)
                self.input_args.append(self.input_names[i])
            elif isinstance(x, (list, tuple)):
                input_name = self.input_names[i]
                for num, item in enumerate(x):
                    if isinstance(item, torch.Tensor):
                        self.input_shapes.append(item.shape)
                        self.input_args.append(f"{input_name}_{num}")
                    else:
                        raise NotImplementedError("Unsupported input type")
            elif isinstance(x, int):
                self.input_shapes.append(None)
                self.input_args.append(self.input_names[i])
        
        # 生成函数参数类型
        # 注意：即使内部使用了量化，函数签名仍然使用 float32
        # quant/dequant 操作会在函数内部的边界处自动插入
        args = []
        for name, shape in zip(self.input_args, self.input_shapes):
            if shape:
                shape_str = ', '.join([str(s) for s in shape])
                if self._is_quantized_input(name):
                    # 量化输入使用 int8 类型
                    dtype_str = self._get_quant_dtype_str(self.quant_act_bits)
                    args.append(f"{name}: {dtype_str}[{shape_str}]")
                else:
                    # 非量化输入使用 float32
                    args.append(f"{name}: float32[{shape_str}]")
            else:
                args.append(f"{name}: int32")
        
        res = ""
        # top-level function
        res += f"def forward({', '.join(args)})".format()
        # outputs
        res += f" -> ({', '.join(self.output)}):\n"
        # subfunctions
        if self.subfunctions:
            res += "\n".join(self.subfunctions) + "\n"
        
        # 处理命名参数声明
        if self.named_params:
            for name, param in self.named_params:
                new_name = name.replace(".", "_")
                shape_str = ', '.join([str(s) for s in param.shape])
                
                # 默认类型
                param_dtype = "float32"
                
                # 检查是否是量化模块的参数
                is_quant, param_suffix = self._is_quant_module_param(name)
                if is_quant:
                    if param_suffix == "weight":
                        param_dtype = self._get_quant_dtype_str(self.quant_weight_bits)
                    elif param_suffix == "bias":
                        param_dtype = self._get_quant_dtype_str(self.quant_bias_bits)
                elif self.enable_quant:
                    # 检查是否是需要量化的特殊 embedding 参数
                    # 这些参数会被传递给量化流程
                    if self._is_quant_embedding_param(name):
                        param_dtype = self._get_quant_dtype_str(self.quant_act_bits)
                
                res += f"    {new_name}: {param_dtype}[{shape_str}] = g_{new_name}\n"
    
        # 处理量化模块的定点参数（仅在启用量化时）
        if self.enable_quant:
            res += self._generate_quantized_param_declarations()
        
        # function body
        for line in self.code:
            res += f"    {line}\n"
        return res

    def _generate_quantized_param_declarations(self):
        """
        为 _process_quantized_params() 注入的定点化参数生成 DSL 声明。
        
        这个函数遍历所有量化模块，为每个模块的量化参数（weight_int, scale_sign, scale_coe, 
        scale_rshift 等）生成类型声明，以便在生成的 DSL 代码中正确引用 global_vars 中的值。
        
        Returns:
            str: 包含所有量化参数声明的字符串
        """
        # 定义需要处理的量化模块类型
        quantized_module_types = (
            QLinear, 
            QConv2d, 
            IntGELU, 
            IntSoftmax, 
            IntLayerNorm, 
            QAdd, 
            QMatMul, 
            QMatMulIsqrtD, 
        )
        
        declarations = []
        
        # 获取配置的位宽类型
        weight_dtype = self._get_quant_dtype_str(self.quant_weight_bits)
        bias_dtype = self._get_quant_dtype_str(self.quant_bias_bits)
        act_dtype = self._get_quant_dtype_str(self.quant_act_bits)
        
        # 遍历所有命名模块
        for module_name, module in self.gm.named_modules():
            if not isinstance(module, quantized_module_types):
                continue
            
            # 模块名转换为变量名格式
            var_prefix = module_name.replace(".", "_")
            
            # ========== 权重相关 ==========
            if hasattr(module, 'weight_int'):
                # weight_int: 使用配置的权重位宽
                if hasattr(module.weight_int, 'shape'):
                    shape = module.weight_int.shape
                    shape_str = ', '.join(str(s) for s in shape)
                    declarations.append(f"    {var_prefix}_weight_int: {weight_dtype}[{shape_str}] = g_{var_prefix}_weight_int")
                
                # weight_scale 的定点表示（sign, coe, rshift）
                if hasattr(module, 'weight_scale'):
                    scale_shape = module.weight_scale.shape
                    if len(scale_shape) == 0:
                        # 标量
                        # NOTE: !!!: Please Notice here if you need to change the type (int8 here) of weight_scale_sign to other type
                        declarations.append(f"    {var_prefix}_weight_scale_sign: int8 = g_{var_prefix}_weight_scale_sign")
                        declarations.append(f"    {var_prefix}_weight_scale_coe: int32 = g_{var_prefix}_weight_scale_coe")
                        declarations.append(f"    {var_prefix}_weight_scale_rshift: int16 = g_{var_prefix}_weight_scale_rshift")
                    else:
                        # 向量/张量
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_weight_scale_sign: int8[{shape_str}] = g_{var_prefix}_weight_scale_sign")
                        declarations.append(f"    {var_prefix}_weight_scale_coe: int32[{shape_str}] = g_{var_prefix}_weight_scale_coe")
                        declarations.append(f"    {var_prefix}_weight_scale_rshift: int16[{shape_str}] = g_{var_prefix}_weight_scale_rshift")
            
            # ========== 偏置相关 ==========
            if hasattr(module, 'bias_int'):
                local_bias_dtype = "int32"
                if hasattr(module.bias_int, 'shape'):
                    bias_shape = module.bias_int.shape
                    if isinstance(module, QLinear):
                        local_bias_dtype = bias_dtype
                        if module.wgt_per_channel:
                            raise NotImplementedError("Per-channel quantization is not supported for QLinear")
                        else:
                            # NOTE: Here is also hard-coded for seq_len usage in bias shape for QLinear which is not classifier.
                            # Previously this was effectively unconditional due to `act_per_token or not act_per_token`.
                            if 'classifier' not in module_name:
                                bias_shape = (self.quant_config.seq_len, bias_shape[-1])
                            else:
                                bias_shape = (1, bias_shape[-1])
                    bias_shape_str = ', '.join(str(s) for s in bias_shape)
                    declarations.append(f"    {var_prefix}_bias_int: {local_bias_dtype}[{bias_shape_str}] = g_{var_prefix}_bias_int")
                
                if hasattr(module, 'bias_scale'):
                    scale_shape = module.bias_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_bias_scale_sign: int8 = g_{var_prefix}_bias_scale_sign")
                        declarations.append(f"    {var_prefix}_bias_scale_coe: int32 = g_{var_prefix}_bias_scale_coe")
                        declarations.append(f"    {var_prefix}_bias_scale_rshift: int16 = g_{var_prefix}_bias_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_bias_scale_sign: int8[{shape_str}] = g_{var_prefix}_bias_scale_sign")
                        declarations.append(f"    {var_prefix}_bias_scale_coe: int32[{shape_str}] = g_{var_prefix}_bias_scale_coe")
                        declarations.append(f"    {var_prefix}_bias_scale_rshift: int16[{shape_str}] = g_{var_prefix}_bias_scale_rshift")
            
            # ========== 激活量化相关的 scale ==========
            scale_names = ['input_scale', 'output_scale', 'fused_scale']
            for scale_name in scale_names:
                if hasattr(module, scale_name):
                    scale_attr = getattr(module, scale_name)
                    if hasattr(scale_attr, 'shape'):
                        scale_shape = scale_attr.shape
                        if len(scale_shape) == 0:
                            # 标量
                            declarations.append(f"    {var_prefix}_{scale_name}_sign: int8 = g_{var_prefix}_{scale_name}_sign")
                            declarations.append(f"    {var_prefix}_{scale_name}_coe: int32 = g_{var_prefix}_{scale_name}_coe")
                            declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16 = g_{var_prefix}_{scale_name}_rshift")
                            
                            # Add inverse for output_scale
                            if scale_name == 'output_scale':
                                declarations.append(f"    {var_prefix}_output_scale_inv_sign: int8 = g_{var_prefix}_output_scale_inv_sign")
                                declarations.append(f"    {var_prefix}_output_scale_inv_coe: int32 = g_{var_prefix}_output_scale_inv_coe")
                                declarations.append(f"    {var_prefix}_output_scale_inv_rshift: int16 = g_{var_prefix}_output_scale_inv_rshift")
                        else:
                            # 向量/张量
                            shape_str = ', '.join(str(s) for s in scale_shape)
                            declarations.append(f"    {var_prefix}_{scale_name}_sign: int8[{shape_str}] = g_{var_prefix}_{scale_name}_sign")
                            declarations.append(f"    {var_prefix}_{scale_name}_coe: int32[{shape_str}] = g_{var_prefix}_{scale_name}_coe")
                            declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16[{shape_str}] = g_{var_prefix}_{scale_name}_rshift")
                            
                            # Add inverse for output_scale
                            if scale_name == 'output_scale':
                                declarations.append(f"    {var_prefix}_output_scale_inv_sign: int8[{shape_str}] = g_{var_prefix}_output_scale_inv_sign")
                                declarations.append(f"    {var_prefix}_output_scale_inv_coe: int32[{shape_str}] = g_{var_prefix}_output_scale_inv_coe")
                                declarations.append(f"    {var_prefix}_output_scale_inv_rshift: int16[{shape_str}] = g_{var_prefix}_output_scale_inv_rshift")
            
            # ========== Zero points（非对称量化）==========
            zero_names = ['input_zero', 'output_zero']
            for zero_name in zero_names:
                if hasattr(module, zero_name):
                    zero_attr = getattr(module, zero_name)
                    if zero_attr is not None and hasattr(zero_attr, 'shape'):
                        zero_shape = zero_attr.shape
                        if len(zero_shape) == 0:
                            declarations.append(f"    {var_prefix}_{zero_name}: {act_dtype} = g_{var_prefix}_{zero_name}")
                        else:
                            shape_str = ', '.join(str(s) for s in zero_shape)
                            declarations.append(f"    {var_prefix}_{zero_name}: {act_dtype}[{shape_str}] = g_{var_prefix}_{zero_name}")
            
            # ========== 特殊处理：IntLayerNorm ==========
            if isinstance(module, IntLayerNorm):
                if hasattr(module, 'layernorm_scale'):
                    scale_shape = module.layernorm_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_layernorm_scale_sign: int8 = g_{var_prefix}_layernorm_scale_sign")
                        declarations.append(f"    {var_prefix}_layernorm_scale_coe: int32 = g_{var_prefix}_layernorm_scale_coe")
                        declarations.append(f"    {var_prefix}_layernorm_scale_rshift: int16 = g_{var_prefix}_layernorm_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_layernorm_scale_sign: int8[{shape_str}] = g_{var_prefix}_layernorm_scale_sign")
                        declarations.append(f"    {var_prefix}_layernorm_scale_coe: int32[{shape_str}] = g_{var_prefix}_layernorm_scale_coe")
                        declarations.append(f"    {var_prefix}_layernorm_scale_rshift: int16[{shape_str}] = g_{var_prefix}_layernorm_scale_rshift")
            
            # ========== 特殊处理：QAdd/QMatMul（双输入）==========
            if isinstance(module, (QAdd, QMatMul, QMatMulIsqrtD)):
                for scale_name in ['x_scale', 'y_scale', 'o_scale']:
                    if hasattr(module, scale_name):
                        scale_attr = getattr(module, scale_name)
                        if hasattr(scale_attr, 'shape'):
                            scale_shape = scale_attr.shape
                            if len(scale_shape) == 0:
                                declarations.append(f"    {var_prefix}_{scale_name}_sign: int8 = g_{var_prefix}_{scale_name}_sign")
                                declarations.append(f"    {var_prefix}_{scale_name}_coe: int32 = g_{var_prefix}_{scale_name}_coe")
                                declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16 = g_{var_prefix}_{scale_name}_rshift")
                                
                                # Add inverse for o_scale
                                if scale_name == 'o_scale':
                                    declarations.append(f"    {var_prefix}_o_scale_inv_sign: int8 = g_{var_prefix}_o_scale_inv_sign")
                                    declarations.append(f"    {var_prefix}_o_scale_inv_coe: int32 = g_{var_prefix}_o_scale_inv_coe")
                                    declarations.append(f"    {var_prefix}_o_scale_inv_rshift: int16 = g_{var_prefix}_o_scale_inv_rshift")
                            else:
                                shape_str = ', '.join(str(s) for s in scale_shape)
                                declarations.append(f"    {var_prefix}_{scale_name}_sign: int8[{shape_str}] = g_{var_prefix}_{scale_name}_sign")
                                declarations.append(f"    {var_prefix}_{scale_name}_coe: int32[{shape_str}] = g_{var_prefix}_{scale_name}_coe")
                                declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16[{shape_str}] = g_{var_prefix}_{scale_name}_rshift")
                                
                                # Add inverse for o_scale
                                if scale_name == 'o_scale':
                                    declarations.append(f"    {var_prefix}_o_scale_inv_sign: int8[{shape_str}] = g_{var_prefix}_o_scale_inv_sign")
                                    declarations.append(f"    {var_prefix}_o_scale_inv_coe: int32[{shape_str}] = g_{var_prefix}_o_scale_inv_coe")
                                    declarations.append(f"    {var_prefix}_o_scale_inv_rshift: int16[{shape_str}] = g_{var_prefix}_o_scale_inv_rshift")
                
                for zero_name in ['x_zero', 'y_zero', 'o_zero']:
                    if hasattr(module, zero_name):
                        zero_attr = getattr(module, zero_name)
                        if zero_attr is not None and hasattr(zero_attr, 'shape'):
                            zero_shape = zero_attr.shape
                            if len(zero_shape) == 0:
                                declarations.append(f"    {var_prefix}_{zero_name}: {act_dtype} = g_{var_prefix}_{zero_name}")
                            else:
                                shape_str = ', '.join(str(s) for s in zero_shape)
                                declarations.append(f"    {var_prefix}_{zero_name}: {act_dtype}[{shape_str}] = g_{var_prefix}_{zero_name}")
            
            # ========== 特殊处理：IntSoftmax ==========
            if isinstance(module, IntSoftmax):
                if hasattr(module, 'softmax_scale'):
                    scale_shape = module.softmax_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_softmax_scale_sign: int8 = g_{var_prefix}_softmax_scale_sign")
                        declarations.append(f"    {var_prefix}_softmax_scale_coe: int32 = g_{var_prefix}_softmax_scale_coe")
                        declarations.append(f"    {var_prefix}_softmax_scale_rshift: int16 = g_{var_prefix}_softmax_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_softmax_scale_sign: int8[{shape_str}] = g_{var_prefix}_softmax_scale_sign")
                        declarations.append(f"    {var_prefix}_softmax_scale_coe: int32[{shape_str}] = g_{var_prefix}_softmax_scale_coe")
                        declarations.append(f"    {var_prefix}_softmax_scale_rshift: int16[{shape_str}] = g_{var_prefix}_softmax_scale_rshift")
            
            # ========== 特殊处理：IntGELU ==========
            if isinstance(module, IntGELU):
                # IntGELU 需要 4 组 scale: input_scale, gelu_scale, fused_scale, output_scale
                # (input_scale 和 output_scale 已经在通用部分声明)
                if hasattr(module, 'gelu_scale'):
                    scale_shape = module.gelu_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_gelu_scale_sign: int8 = g_{var_prefix}_gelu_scale_sign")
                        declarations.append(f"    {var_prefix}_gelu_scale_coe: int32 = g_{var_prefix}_gelu_scale_coe")
                        declarations.append(f"    {var_prefix}_gelu_scale_rshift: int16 = g_{var_prefix}_gelu_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_gelu_scale_sign: int8[{shape_str}] = g_{var_prefix}_gelu_scale_sign")
                        declarations.append(f"    {var_prefix}_gelu_scale_coe: int32[{shape_str}] = g_{var_prefix}_gelu_scale_coe")
                        declarations.append(f"    {var_prefix}_gelu_scale_rshift: int16[{shape_str}] = g_{var_prefix}_gelu_scale_rshift")

        # 返回所有声明，每个声明一行
        return '\n'.join(declarations) + '\n' if declarations else ''

    def __call__(self, node):
        # 在生成当前节点代码之前，检查是否需要在边界处插入 quant/dequant
        # 处理 call_module 和 call_function 节点
        if self.enable_quant and node.op in {'call_module', 'call_function', 'call_method'}:
            self._insert_boundary_conversions(node)
        
        method = getattr(self, "build_" + node.op)
        ret = method(node)
        if ret:
            self.code.append(ret)
        return ret
    
    def _insert_boundary_conversions(self, node):
        """为当前节点插入所需的边界转换代码（quant/dequant）"""
        for i, arg in enumerate(node.args):
            if not isinstance(arg, fx.Node):
                continue
            
            key = (arg.name, node.name, i)
            if key in self._boundary_conversions:
                conv_info = self._boundary_conversions[key]
                
                # 检查是否真的需要类型转换
                # 例如：int8 -> int8 就不需要 quant，float32 -> float32 就不需要 dequant
                needs_conversion = self._check_if_conversion_needed(arg, node, conv_info)
                
                if needs_conversion:
                    result = self._generate_boundary_conversion(conv_info, arg.name, node.name, i)
                    if result:
                        new_var_name, code = result
                        # 插入转换代码
                        self.code.append(code)
                        # 记录变量替换（后续的 build 方法需要使用新变量名）
                        if not hasattr(self, '_var_substitutions'):
                            self._var_substitutions = {}
                        self._var_substitutions[(node.name, i)] = new_var_name

    def _check_if_conversion_needed(self, producer_node, consumer_node, conv_info):
        """检查是否真的需要类型转换
        
        Args:
            producer_node: 生产者节点（输入来源）
            consumer_node: 消费者节点（当前节点）
            conv_info: 转换信息字典
        
        Returns:
            bool: 是否需要转换
        """
        # 获取生产者和消费者的模块
        modules_dict = dict(self.gm.named_modules())
        quantized_module_types = (
            QLinear, QConv2d, IntGELU, IntSoftmax, 
            IntLayerNorm, QAdd, QMatMul, QMatMulIsqrtD,
        )
        
        producer_module = None
        consumer_module = None
        
        if producer_node.op == 'call_module':
            producer_module = modules_dict.get(producer_node.target)
        if consumer_node.op == 'call_module':
            consumer_module = modules_dict.get(consumer_node.target)
        
        conv_type = conv_info['type']
        
        if conv_type == 'quant':
            # quant 转换：float -> int
            # 如果生产者已经输出 int，就不需要 quant
            if producer_module is not None and isinstance(producer_module, quantized_module_types):
                # 生产者是量化模块，输出已经是 int，不需要 quant
                return False
            # 其他情况需要 quant（float -> int）
            return True
        
        elif conv_type == 'dequant':
            # dequant 转换：int -> float
            # 如果消费者也接受 int 输入，就不需要 dequant
            if consumer_module is not None and isinstance(consumer_module, quantized_module_types):
                # 消费者是量化模块，可以接受 int 输入，不需要 dequant
                return False
            # 其他情况需要 dequant（int -> float）
            return True
        
        return True
    
    def _get_substituted_var_name(self, node, arg, arg_index):
        """获取可能被替换的变量名（用于边界转换后的变量）"""
        if hasattr(self, '_var_substitutions'):
            key = (node.name, arg_index)
            if key in self._var_substitutions:
                return self._var_substitutions[key]
        return get_var_name(arg)

    def get_module(self, name):
        return dict(self.gm.named_modules())[name]

    def build_placeholder(self, node):
        self.input_names.append(node.name)

    def build_getattr(self, node):
        pass

    def build_get_attr(self, node):
        pass

    def build_call_module(self, node):
        module = self.get_module(node.target)
        op = {
            torch.nn.Linear: "linear",
            torch.nn.Dropout: "identity",
            torch.nn.GELU: "gelu",
            torch.nn.Tanh: "tanh",
            torch.nn.LayerNorm: "layernorm",
            torch.nn.Conv2d: "conv2d",
        }.get(type(module), None)
        if self.leaf_modules:
            for leaf_module in self.leaf_modules:
                if isinstance(module, leaf_module):
                    if module.__class__.__name__ == "ViTGetFirstToken":
                        return getattr(self, f"build_{module.__class__.__name__}")(node, module.shape)
                    elif module.__class__.__name__ == "QConv2d":
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
                    elif module.__class__.__name__ == "QLinear":
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
                    elif module.__class__.__name__ == "ViTTokenExpand":
                        return getattr(self, f"build_{module.__class__.__name__}")(node, module.token_shape)
                    elif module.__class__.__name__ == "IntSoftmax":
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
                    elif module.__class__.__name__ == "IntLayerNorm":
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
                    elif module.__class__.__name__ == "IntGELU":
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
                    elif module.__class__.__name__ == "QAdd":
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
                    elif module.__class__.__name__ == "QMatMul":
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
                    elif module.__class__.__name__ == "QMatMulIsqrtD":
                        # Do not expose qmatmul_isqrtd to downstream; the sqrt(dim)
                        # effect is fused into scales inside the quant module.
                        return self.build_QMatMul(node)
        if op is None:
            raise NotImplementedError("Unsupported module")
        if op == "linear":
            bias = True if module.bias is not None else None
            return getattr(self, "build_linear")(node, bias)
        if op == "conv2d":
            bias = True if module.bias is not None else None
            return getattr(self, "build_conv2d")(node, module.stride, bias)
        return getattr(self, f"build_{op}")(node)

    def build_call_function(self, node):
        # map known function targets to builder method names
        mapping = {
            operator.add: "add",
            operator.sub: "sub",
            operator.mul: "mul",
            operator.truediv: "div",
            operator.neg: "neg",
            operator.matmul: "matmul",
            operator.getitem: "getitem",
            operator.pow: "pow",
            torch.matmul: "matmul",
            torch.ones: "ones",
            torch.zeros: "zeros",
            math.sqrt: "sqrt",
            F.softmax: "softmax",
            F.linear: "linear",
            F.gelu: "gelu",
            F.relu: "relu",
            F.dropout: "identity",
            torch.tril: "tril",
            torch.cat: "concat",
            torch.round: "round",
            torch.floor: "floor",
            torch.clamp: "clamp",
            torch.pow: "pow",
            torch.max: "max",
            torch.sum: "sum",
            torch.matmul: "matmul",
        }

        # Only build nodes that have tensor metadata (shape/dtype).
        if "tensor_meta" not in node.meta:
            return None

        op_name = mapping.get(node.target)
        if op_name is None:
            # Actionable error with node context
            raise NotImplementedError(
                f"Unsupported call_function target for node '{node.name}': "
                f"target={node.target!r}, op={node.op!r}. "
                "Add mapping in build_call_function or implement build_<op>."
            )

        build_fn = getattr(self, f"build_{op_name}", None)
        if build_fn is None:
            raise NotImplementedError(
                f"Builder for operation '{op_name}' not implemented "
                f"(node='{node.name}', target={node.target!r})."
            )

        return build_fn(node)

    def build_call_method(self, node):
        if node.target == "contiguous":
            return self.build_identity(node)
        # Only nodes with shape need to be built.
        return (
            getattr(self, f"build_{node.target}")(node)
            if "tensor_meta" in node.meta
            else None
        )

    def append_output(self, output, output_name=None):
        """
        添加输出类型声明。
        
        Args:
            output: TensorMetadata，包含shape和dtype
            output_name: 输出节点名称，用于检查是否是量化输出
        """
        shape = str(list(output.shape))
        
        # 检查是否是量化模块的输出
        if output_name and self._is_quantized_output(output_name):
            # 量化输出使用 int8 类型
            dtype = self._get_quant_dtype_str(self.quant_act_bits)
        else:
            dtype = str(output.dtype)[6:]
        
        self.output.append(dtype + shape)

    def build_output(self, node):
        # 获取返回的变量名称（用于检查是否是量化输出）
        raw_name = get_var_name(node.args[0])
        if isinstance(raw_name, dict):
            output_names = list(raw_name.values())
        elif isinstance(raw_name, (list, tuple)):
            output_names = list(raw_name)
        else:
            output_names = [raw_name]
        
        # 将输出名称列表规范化
        output_names_flat = []
        for n in output_names:
            if isinstance(n, fx.Node):
                output_names_flat.append(n.name)
            elif isinstance(n, str):
                output_names_flat.append(n)
            else:
                output_names_flat.append(str(n))
        
        output_idx = [0]  # 使用列表以便在嵌套函数中修改
        
        def get_next_output_name():
            if output_idx[0] < len(output_names_flat):
                name = output_names_flat[output_idx[0]]
                output_idx[0] += 1
                return name
            return None
        
        if isinstance(node.meta["tensor_meta"], TensorMetadata):
            self.append_output(node.meta["tensor_meta"], get_next_output_name())
        elif isinstance(node.meta["tensor_meta"], (list, tuple)):
            for output in node.meta["tensor_meta"]:
                if isinstance(output, TensorMetadata):
                    self.append_output(output, get_next_output_name())
                elif isinstance(output, (list, tuple)):
                    for item in output:
                        if isinstance(item, TensorMetadata):
                            self.append_output(item, get_next_output_name())
                elif isinstance(output, dict):
                    for item in output.values():
                        if isinstance(item, TensorMetadata):
                            self.append_output(item, get_next_output_name())
                        else:
                            raise NotImplementedError("Unsupported output type")
        elif isinstance(node.meta["tensor_meta"], dict):
            for output in node.meta["tensor_meta"].values():
                if isinstance(output, TensorMetadata):
                    self.append_output(output, get_next_output_name())
        
        # 检查是否需要在返回前插入 dequant（量化输出 -> float32 返回值）
        # 因为函数签名总是 float32，如果返回的是量化值，需要 dequant
        return_vars = []
        if self.enable_quant and isinstance(node.args[0], fx.Node):
            return_arg = node.args[0]
            # 检查返回值是否来自量化模块
            if return_arg.op == 'call_module':
                modules_dict = dict(self.gm.named_modules())
                producer_module = modules_dict.get(return_arg.target)
                quantized_module_types = (
                    QLinear, QConv2d, IntGELU, IntSoftmax, 
                    IntLayerNorm, QAdd, QMatMul, QMatMulIsqrtD,
                )
                if producer_module is not None and isinstance(producer_module, quantized_module_types):
                    # 需要在返回前插入 dequant
                    dequant_var_name = f"{return_arg.name}_dequant_return"
                    quant_mode = self._get_quant_mode_from_module(producer_module)
                    scale_name = self._get_output_scale_name(producer_module)
                    zero_name = self._get_output_zero_name(producer_module)
                    
                    # 生成 dequant 代码
                    dequant_code = self.generate_dequant_call(
                        dequant_var_name,
                        return_arg.name,
                        quant_mode,
                        f"{return_arg.target.replace('.', '_')}_{scale_name}_sign",
                        f"{return_arg.target.replace('.', '_')}_{scale_name}_coe",
                        f"{return_arg.target.replace('.', '_')}_{scale_name}_rshift",
                        f"{return_arg.target.replace('.', '_')}_{zero_name}" if hasattr(producer_module, zero_name) and getattr(producer_module, zero_name) is not None else None
                    )
                    self.code.append(dequant_code)
                    return_vars.append(dequant_var_name)
        
        # Unwrap all outputs and return them
        if not return_vars:
            name = get_var_name(node.args[0])
            if isinstance(name, dict):
                name = list(name.values())
            return_name = (
                str(name)
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
            )
        else:
            return_name = ', '.join(return_vars)
        
        if return_name.endswith(","):
            return_name = return_name[:-1]
        return f"return ({return_name})"

    def build_getitem(self, node):
        inp = get_var_name(node.args[0])
        index = node.args[1]
        return f"{node.name} = {inp}_{index}"

    def build_add(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} + {rhs}"

    def build_sub(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} - {rhs}"

    def build_mul(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} * {rhs}"

    def build_matmul(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = dsl.matmul({lhs}, {rhs})"

    def build_div(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} / {rhs}"
    
    def build_round(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = round({inp})"
    def build_floor(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.floor({inp})"

    def build_clamp(self, node):
        inp = get_var_name(node.args[0])
        min_val = node.kwargs.get("min", None)
        max_val = node.kwargs.get("max", None)

        if min_val is None and len(node.args) > 1:
            candidate = node.args[1]
            if len(node.args) > 2:
                min_val = candidate
            else:
                max_val = candidate

        if max_val is None and len(node.args) > 2:
            max_val = node.args[2]

        def fmt_val(v):
            if isinstance(v, fx.Node):
                return get_var_name(v)
            if v is None:
                return "None"
            return repr(v)

        min_s = fmt_val(min_val)
        max_s = fmt_val(max_val)

        if min_val is None and max_val is None:
            return f"{node.name} = {inp}"
        if min_val is None:
            return f"{node.name} = dsl.clamp_max({inp}, {max_s})"
        if max_val is None:
            return f"{node.name} = dsl.clamp({inp}, {min_s})"
        return f"{node.name} = dsl.clamp({inp}, {min_s}, {max_s})"

    def build_clamp_max_(self, node):
        inp = get_var_name(node.args[0])
        max_val = node.kwargs.get("max", None)
        return f"{node.name} = dsl.clamp_max({inp}, {max_val})"

    def build_neg(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = -{inp}"

    def build_pow(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = dsl.power({lhs}, {rhs})"

    def build_mean(self, node):
        inp = get_var_name(node.args[0])
        dim = node.kwargs.get("axis", None)
        if dim is None:
            dim = node.kwargs.get("dim", None)
        keepdim = node.kwargs.get("keepdim", False)
        return f"{node.name} = dsl.mean({inp}, {dim}, {keepdim})"

    def build_sum(self, node):
        inp = get_var_name(node.args[0])
        dim = node.kwargs.get("axis", None)
        if dim is None:
            dim = node.kwargs.get("dim", None)
        keepdim = node.kwargs.get("keepdim", False)
        return f"{node.name} = dsl.sum({inp}, {dim}, {keepdim})"

    def build_max(self, node):
        inp = get_var_name(node.args[0])
        dim = node.kwargs.get("axis", None)
        if dim is None:
            dim = node.kwargs.get("dim", None)
        keepdim = node.kwargs.get("keepdim", False)
        if dim is None:
            return f"{node.name} = dsl.max({inp})"
        return f"{node.name} = dsl.max({inp}, {dim}, {keepdim})"

    def build_softmax(self, node):
        if node.kwargs.get("dim") != -1:
            raise NotImplementedError("Only support softmax on the last dimension")
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.softmax({inp})"

    def build_relu(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.relu({inp})"

    def build_linear(self, node, bias):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        if bias:
            bias = get_var_name(target_name + "_bias")
            return f"{node.name} = dsl.linear({inp}, {weight}, {bias})"
        return f"{node.name} = dsl.linear({inp}, {weight})"
    
    def build_conv2d(self, node, stride, bias):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        if bias:
            bias = get_var_name(target_name + "_bias")
            return f"{node.name} = dsl.conv2d({inp}, {weight}, {stride}, {bias})"
        return f"{node.name} = dsl.conv2d({inp}, {weight}, {stride})"

    def build_gelu(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.gelu({inp})"
    
    def build_tanh(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.tanh({inp})"

    def build_layernorm(self, node):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        bias = get_var_name(target_name + "_bias")
        return f"{node.name} = dsl.layernorm({inp}, {weight}, {bias})"

    def build_view(self, node):
        inp = get_var_name(node.args[0])
        shape = tuple(node.meta["tensor_meta"].shape)
        return f"{node.name} = dsl.view({inp}, {shape})"
    
    def build_expand(self, node):
        inp = get_var_name(node.args[0])
        shape = tuple(node.meta["tensor_meta"].shape)
        return f"{node.name} = dsl.expand({inp}, {shape})"

    def build_reshape(self, node):
        return self.build_view(node)

    def build_permute(self, node):
        inp = get_var_name(node.args[0])
        permutation = node.args[1:]
        return f"{node.name} = dsl.transpose({inp}, {permutation})"

    def build_transpose(self, node):
        # PyTorch only supports transposing two dimensions,
        # https://pytorch.org/docs/stable/generated/torch.transpose.html
        inp = get_var_name(node.args[0])
        shape_len = len(node.meta["tensor_meta"].shape)
        sorted_args = sorted(
            [
                node.args[1] if node.args[1] >= 0 else node.args[1] + shape_len,
                node.args[2] if node.args[2] >= 0 else node.args[2] + shape_len,
            ]
        )
        permutation = list(range(shape_len))
        permutation[sorted_args[0]] = sorted_args[1]
        permutation[sorted_args[1]] = sorted_args[0]
        return f"{node.name} = dsl.transpose({inp}, {tuple(permutation)})"

    def build_identity(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = {inp}"

    def build_ones(self, node):
        shape = tuple(node.meta["tensor_meta"].shape)
        dtype = node.meta["tensor_meta"].dtype
        if str(dtype).startswith("torch."):
            dtype = str(dtype)[6:]
        return f"{node.name} = dsl.ones({shape}, dtype={dtype})"

    def build_zeros(self, node):
        shape = tuple(node.meta["tensor_meta"].shape)
        dtype = node.meta["tensor_meta"].dtype
        if str(dtype).startswith("torch."):
            dtype = str(dtype)[6:]
        return f"{node.name} = dsl.zeros({shape}, dtype={dtype})"

    def build_tril(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.tril({inp})"

    def build_concat(self, node):
        shape_len = len(node.meta["tensor_meta"].shape)
        tensor_A = get_var_name(node.args[0][0])
        tensor_B = get_var_name(node.args[0][1])
        dim = node.kwargs["dim"] + (node.kwargs["dim"] < 0) * shape_len
        return f"{node.name} = dsl.concat({tensor_A}, {tensor_B}, axis={dim})"
    
    # Deprecated
    # def build_ViTGetFirstToken(self, node, shape):
    #     shape = (self.example_inputs[0].shape[0], shape[1], shape[2])
    #     # 根据输入节点的实际类型选择数据类型
    #     # 检查输入参数的类型（从 _node_output_types 查询）
    #     dtype_str = "float32"  # 默认值
    #     if self.enable_quant and hasattr(self, '_node_output_types'):
    #         # 查找输入节点的类型
    #         for arg in node.args:
    #             if isinstance(arg, fx.Node) and arg.name in self._node_output_types:
    #                 output_type = self._node_output_types[arg.name]
    #                 if output_type == 'int8':
    #                     dtype_str = self._get_quant_dtype_str(self.quant_act_bits)
    #                 break
    #     # 根据数据类型选择对应的 lib 函数
    #     lib_func = ViTGetFirstToken_int8_lib if dtype_str == f"int{self.quant_act_bits}" else ViTGetFirstToken_float32_lib
    #     src = inspect.getsource(lib_func(*shape))
    #     src = (
    #         src.replace("s_0", str(shape[0]))
    #         .replace("s_1", str(shape[1]))
    #         .replace("s_2", str(shape[2]))
    #     )
    #     if src not in self.subfunctions:
    #         self.subfunctions.append(src)
    #     return f"{node.name} = ViTGetFirstToken({', '.join([get_var_name(arg) for arg in node.args])})"
    
    def build_ViTGetFirstToken(self, node, shape):
        """构建 ViTGetFirstToken 的 DSL 调用
        
        功能：从 [B, L, D] 提取第一个 token 得到 [B, 1, D]
        
        Args:
            node: 节点信息
            shape: 原始 token 形状 (不含 batch)
        
        Returns:
            DSL 调用字符串
        """
        # 获取输入变量名
        inp = get_var_name(node.args[0])
        
        # 直接调用 dsl.vit_get_first_token，不需要子函数
        return f"{node.name} = dsl.vit_get_first_token({inp})"

    def build_ViTTokenExpand(self, node, shape):
        shape = (self.example_inputs[0].shape[0], shape[1], shape[2])
        # 根据输入节点的实际类型选择数据类型
        # 检查输入参数的类型（从 _node_output_types 查询）
        dtype_str = "float32"  # 默认值
        if self.enable_quant and hasattr(self, '_node_output_types'):
            # 查找输入节点的类型
            for arg in node.args:
                if isinstance(arg, fx.Node) and arg.name in self._node_output_types:
                    output_type = self._node_output_types[arg.name]
                    if output_type == 'int8':
                        dtype_str = self._get_quant_dtype_str(self.quant_act_bits)
                    break
        # 根据数据类型选择对应的 lib 函数
        lib_func = ViTTokenExpand_int8_lib if dtype_str == f"int{self.quant_act_bits}" else ViTTokenExpand_float32_lib
        src = inspect.getsource(lib_func(*shape))
        src = (
            src.replace("s_0", str(shape[0]))
            .replace("s_1", str(shape[1]))
            .replace("s_2", str(shape[2]))
            .replace("ViTTokenExpand", f"ViTTokenExpand_{node.name}")
        )
        if src not in self.subfunctions:
            self.subfunctions.append(src)
        return f"{node.name} = ViTTokenExpand_{node.name}({', '.join([get_var_name(arg) for arg in node.args][:1])})"

    def build_CoreAttention(self, node):
        shape = tuple(self.example_inputs[1][0].shape)
        src = inspect.getsource(CoreAttention_lib(*shape))
        src = (
            src.replace("s_0", str(shape[0]))
            .replace("s_1", str(shape[1]))
            .replace("s_2", str(shape[2]))
            .replace("s_3", str(shape[3]))
        )

        if src not in self.subfunctions:
            self.subfunctions.append(src)
        return f"{node.name} = CoreAttention({', '.join([get_var_name(arg) for arg in node.args])})"

    def build_KVCache(self, node):
        shape = tuple(node.meta["tensor_meta"][0])
        src = inspect.getsource(KVCache_lib(*shape))
        src = (
            src.replace("s_0", str(shape[0]))
            .replace("s_1", str(shape[1]))
            .replace("s_2", str(shape[2]))
            .replace("s_3", str(shape[3]))
        )

        if src not in self.subfunctions:
            self.subfunctions.append(src)
        return f"{node.name} = KVCache({', '.join([get_var_name(arg) for arg in node.args])})"
    
    def build_QConv2d(self, node):
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp = self._get_substituted_var_name(node, node.args[0], 0)
        weight = get_var_name(target_name + "_weight_int")
        stride = tuple(module.stride) if isinstance(module.stride, (list, tuple)) else (module.stride, module.stride)
        
        # fused_scale 的拆分参数
        fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
        fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
        fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
        
        # input_scale 的拆分参数
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        # output_scale 的拆分参数
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
        # output_scale_inv 的拆分参数
        output_scale_inv_sign = get_var_name(target_name + "_output_scale_inv_sign")
        output_scale_inv_coe = get_var_name(target_name + "_output_scale_inv_coe")
        output_scale_inv_rshift = get_var_name(target_name + "_output_scale_inv_rshift")
        
        # weight_scale 的拆分参数
        weight_scale_sign = get_var_name(target_name + "_weight_scale_sign")
        weight_scale_coe = get_var_name(target_name + "_weight_scale_coe")
        weight_scale_rshift = get_var_name(target_name + "_weight_scale_rshift")
        
        params = [
            inp, weight, stride,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            input_scale_sign, input_scale_coe, input_scale_rshift,
            output_scale_sign, output_scale_coe, output_scale_rshift,
            output_scale_inv_sign, output_scale_inv_coe, output_scale_inv_rshift,
            weight_scale_sign, weight_scale_coe, weight_scale_rshift
        ]
        
        kwargs = []
        
        if module.bias is not None:
            # bias_scale 的拆分参数
            bias_scale_sign = get_var_name(target_name + "_bias_scale_sign")
            bias_scale_coe = get_var_name(target_name + "_bias_scale_coe")
            bias_scale_rshift = get_var_name(target_name + "_bias_scale_rshift")
            kwargs.append(f"bscl_sign={bias_scale_sign}")
            kwargs.append(f"bscl_coe={bias_scale_coe}")
            kwargs.append(f"bscl_rshift={bias_scale_rshift}")
            
            bias = get_var_name(target_name + "_bias_int")
            kwargs.append(f"bias={bias}")

        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            kwargs.append(f"izr={input_zero}")
        
        if hasattr(module, 'output_zero') and module.output_zero is not None:
            output_zero = get_var_name(target_name + "_output_zero")
            kwargs.append(f"ozr={output_zero}")
        
        # 组装参数字符串
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
        
        return f"{node.name} = dsl.qconv2d({params_str})"

    def build_QLinear(self, node):
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp = self._get_substituted_var_name(node, node.args[0], 0)
        weight = get_var_name(target_name + "_weight_int")
        
        # fused_scale 的拆分参数
        fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
        fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
        fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
        
        # input_scale 的拆分参数
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        # output_scale 的拆分参数
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
        # output_scale_inv 的拆分参数
        output_scale_inv_sign = get_var_name(target_name + "_output_scale_inv_sign")
        output_scale_inv_coe = get_var_name(target_name + "_output_scale_inv_coe")
        output_scale_inv_rshift = get_var_name(target_name + "_output_scale_inv_rshift")
        
        # weight_scale 的拆分参数
        weight_scale_sign = get_var_name(target_name + "_weight_scale_sign")
        weight_scale_coe = get_var_name(target_name + "_weight_scale_coe")
        weight_scale_rshift = get_var_name(target_name + "_weight_scale_rshift")
        
        params = [
            inp, weight,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            input_scale_sign, input_scale_coe, input_scale_rshift,
            output_scale_sign, output_scale_coe, output_scale_rshift,
            output_scale_inv_sign, output_scale_inv_coe, output_scale_inv_rshift,
            weight_scale_sign, weight_scale_coe, weight_scale_rshift
        ]
        
        kwargs = []
        
        if module.bias is not None:
            # bias_scale 的拆分参数
            bias_scale_sign = get_var_name(target_name + "_bias_scale_sign")
            bias_scale_coe = get_var_name(target_name + "_bias_scale_coe")
            bias_scale_rshift = get_var_name(target_name + "_bias_scale_rshift")
            kwargs.append(f"bscl_sign={bias_scale_sign}")
            kwargs.append(f"bscl_coe={bias_scale_coe}")
            kwargs.append(f"bscl_rshift={bias_scale_rshift}")
            
            bias = get_var_name(target_name + "_bias_int")
            kwargs.append(f"bias={bias}")

        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            kwargs.append(f"izr={input_zero}")
        
        if hasattr(module, 'output_zero') and module.output_zero is not None:
            output_zero = get_var_name(target_name + "_output_zero")
            kwargs.append(f"ozr={output_zero}")

        name = "unknown"
        names = node.name.split('_')
        if names[-1] in ('q', 'k', 'v'):
            name = "qkvgemm.proj_" + names[-1]
        elif names[-1] in ('out',):
            name = "attngemm.proj"
        elif names[-1] in ('fc1', 'fc2'):
            name = "ffn." + names[-1]
        elif names[-1] in ('dense',):
            name = "classifier.dense"
        kwargs.append(f"layer_type=\"{name}\"")
        
        # 组装参数字符串
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
        
        return f"{node.name} = dsl.qlinear({params_str})"
    
    def build_QAdd(self, node):
        """构建 QAdd 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp1 = self._get_substituted_var_name(node, node.args[0], 0)
        inp2 = self._get_substituted_var_name(node, node.args[1], 1)
        
        # 获取两个输入和一个输出的 scale 参数
        x_scale_sign = get_var_name(target_name + "_x_scale_sign")
        x_scale_coe = get_var_name(target_name + "_x_scale_coe")
        x_scale_rshift = get_var_name(target_name + "_x_scale_rshift")
        
        y_scale_sign = get_var_name(target_name + "_y_scale_sign")
        y_scale_coe = get_var_name(target_name + "_y_scale_coe")
        y_scale_rshift = get_var_name(target_name + "_y_scale_rshift")
        
        o_scale_sign = get_var_name(target_name + "_o_scale_sign")
        o_scale_coe = get_var_name(target_name + "_o_scale_coe")
        o_scale_rshift = get_var_name(target_name + "_o_scale_rshift")
        
        o_scale_inv_sign = get_var_name(target_name + "_o_scale_inv_sign")
        o_scale_inv_coe = get_var_name(target_name + "_o_scale_inv_coe")
        o_scale_inv_rshift = get_var_name(target_name + "_o_scale_inv_rshift")
        
        params = [
            inp1, inp2,
            x_scale_sign, x_scale_coe, x_scale_rshift,
            y_scale_sign, y_scale_coe, y_scale_rshift,
            o_scale_sign, o_scale_coe, o_scale_rshift,
            o_scale_inv_sign, o_scale_inv_coe, o_scale_inv_rshift
        ]
        
        kwargs = []
        # zero points
        if hasattr(module, 'x_zero') and module.x_zero is not None:
            x_zero = get_var_name(target_name + "_x_zero")
            y_zero = get_var_name(target_name + "_y_zero")
            o_zero = get_var_name(target_name + "_o_zero")
            kwargs.append(f"x_zero={x_zero}")
            kwargs.append(f"y_zero={y_zero}")
            kwargs.append(f"o_zero={o_zero}")
    
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
            
        return f"{node.name} = dsl.qadd({params_str})"
    
    def build_IntSoftmax(self, node):
        """构建 IntSoftmax 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp = self._get_substituted_var_name(node, node.args[0], 0)
        
        # 获取 scale 参数
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
        output_scale_inv_sign = get_var_name(target_name + "_output_scale_inv_sign")
        output_scale_inv_coe = get_var_name(target_name + "_output_scale_inv_coe")
        output_scale_inv_rshift = get_var_name(target_name + "_output_scale_inv_rshift")
        
        softmax_scale_sign = get_var_name(target_name + "_softmax_scale_sign")
        softmax_scale_coe = get_var_name(target_name + "_softmax_scale_coe")
        softmax_scale_rshift = get_var_name(target_name + "_softmax_scale_rshift")
        
        params = [
            inp,
            input_scale_sign, input_scale_coe, input_scale_rshift,
            softmax_scale_sign, softmax_scale_coe, softmax_scale_rshift,
            output_scale_sign, output_scale_coe, output_scale_rshift,
            output_scale_inv_sign, output_scale_inv_coe, output_scale_inv_rshift
        ]
        
        kwargs = []
        if hasattr(module, 'fused_scale'):
            fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
            fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
            fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
            kwargs.append(f"fused_scale_sign={fused_scale_sign}")
            kwargs.append(f"fused_scale_coe={fused_scale_coe}")
            kwargs.append(f"fused_scale_rshift={fused_scale_rshift}")
        
        # zero points (如果是非对称量化)
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            output_zero = get_var_name(target_name + "_output_zero")
            kwargs.append(f"input_zero={input_zero}")
            kwargs.append(f"output_zero={output_zero}")
        
        # 构建 DSL 调用
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
            
        return f"{node.name} = dsl.int_softmax({params_str})"
    
    def build_IntLayerNorm(self, node):
        """构建 IntLayerNorm 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp = self._get_substituted_var_name(node, node.args[0], 0)
        
        # bias_int
        bias_int = get_var_name(target_name + "_bias_int")
        
        # 获取 scale 参数
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
        output_scale_inv_sign = get_var_name(target_name + "_output_scale_inv_sign")
        output_scale_inv_coe = get_var_name(target_name + "_output_scale_inv_coe")
        output_scale_inv_rshift = get_var_name(target_name + "_output_scale_inv_rshift")
        
        layernorm_scale_sign = get_var_name(target_name + "_layernorm_scale_sign")
        layernorm_scale_coe = get_var_name(target_name + "_layernorm_scale_coe")
        layernorm_scale_rshift = get_var_name(target_name + "_layernorm_scale_rshift")
        
        bias_scale_sign = get_var_name(target_name + "_bias_scale_sign")
        bias_scale_coe = get_var_name(target_name + "_bias_scale_coe")
        bias_scale_rshift = get_var_name(target_name + "_bias_scale_rshift")
        
        # fused_scale (用于最后的重量化)
        if hasattr(module, 'fused_scale'):
            fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
            fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
            fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
        else:
            raise ValueError("IntLayerNorm requires fused_scale")
            
        params = [
            inp, bias_int,
            input_scale_sign, input_scale_coe, input_scale_rshift,
            layernorm_scale_sign, layernorm_scale_coe, layernorm_scale_rshift,
            bias_scale_sign, bias_scale_coe, bias_scale_rshift,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            output_scale_sign, output_scale_coe, output_scale_rshift,
            output_scale_inv_sign, output_scale_inv_coe, output_scale_inv_rshift
        ]
        
        kwargs = []
        # zero points
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            output_zero = get_var_name(target_name + "_output_zero")
            kwargs.append(f"input_zero={input_zero}")
            kwargs.append(f"output_zero={output_zero}")
        
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
        
        return f"{node.name} = dsl.int_layernorm({params_str})"
    
    def build_IntGELU(self, node):
        """构建 IntGELU 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp = self._get_substituted_var_name(node, node.args[0], 0)
        
        # 获取 4 组 scale 参数: input, gelu, fused, output
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        gelu_scale_sign = get_var_name(target_name + "_gelu_scale_sign")
        gelu_scale_coe = get_var_name(target_name + "_gelu_scale_coe")
        gelu_scale_rshift = get_var_name(target_name + "_gelu_scale_rshift")
        
        fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
        fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
        fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
        
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
        output_scale_inv_sign = get_var_name(target_name + "_output_scale_inv_sign")
        output_scale_inv_coe = get_var_name(target_name + "_output_scale_inv_coe")
        output_scale_inv_rshift = get_var_name(target_name + "_output_scale_inv_rshift")
        
        params = [
            inp,
            input_scale_sign, input_scale_coe, input_scale_rshift,
            gelu_scale_sign, gelu_scale_coe, gelu_scale_rshift,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            output_scale_sign, output_scale_coe, output_scale_rshift,
            output_scale_inv_sign, output_scale_inv_coe, output_scale_inv_rshift
        ]
        
        kwargs = []
        # zero points (可选)
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            output_zero = get_var_name(target_name + "_output_zero")
            kwargs.append(f"input_zero={input_zero}")
            kwargs.append(f"output_zero={output_zero}")
            
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
            
        return f"{node.name} = dsl.int_gelu({params_str})"
    def build_QMatMul(self, node):
        """构建 QMatMul 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp1 = self._get_substituted_var_name(node, node.args[0], 0)
        inp2 = self._get_substituted_var_name(node, node.args[1], 1)
        
        # 获取两个输入和一个输出的 scale 参数
        x_scale_sign = get_var_name(target_name + "_x_scale_sign")
        x_scale_coe = get_var_name(target_name + "_x_scale_coe")
        x_scale_rshift = get_var_name(target_name + "_x_scale_rshift")
        
        y_scale_sign = get_var_name(target_name + "_y_scale_sign")
        y_scale_coe = get_var_name(target_name + "_y_scale_coe")
        y_scale_rshift = get_var_name(target_name + "_y_scale_rshift")

        fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
        fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
        fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
        
        o_scale_sign = get_var_name(target_name + "_o_scale_sign")
        o_scale_coe = get_var_name(target_name + "_o_scale_coe")
        o_scale_rshift = get_var_name(target_name + "_o_scale_rshift")
        
        o_scale_inv_sign = get_var_name(target_name + "_o_scale_inv_sign")
        o_scale_inv_coe = get_var_name(target_name + "_o_scale_inv_coe")
        o_scale_inv_rshift = get_var_name(target_name + "_o_scale_inv_rshift")
        
        params = [
            inp1, inp2,
            x_scale_sign, x_scale_coe, x_scale_rshift,
            y_scale_sign, y_scale_coe, y_scale_rshift,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            o_scale_sign, o_scale_coe, o_scale_rshift,
            o_scale_inv_sign, o_scale_inv_coe, o_scale_inv_rshift
        ]
        
        kwargs = []

        # layer_type: 用于后续 Vivado 侧的 layout/优化决策
        # 约定：QMatMul(softmax@V) -> "attn.SV"，QMatMulIsqrtD(Q@K^T/sqrt(d)) -> "attn.QK"
        # 这里对 QMatMul 默认给 SV；若从 node 名称中能识别 matmul2，则更稳。
        layer_type = "attn.SV"
        node_name_l = (node.name or "").lower()
        node_tgt_l = (getattr(node, "target", "") or "").lower()
        if "matmul2" in node_name_l or "matmul2" in node_tgt_l:
            layer_type = "attn.SV"
        elif "matmul1" in node_name_l or "matmul1" in node_tgt_l:
            # 兼容：如果上游误把 matmul1 也走到 qmatmul，这里避免标错
            layer_type = "attn.QK"
        kwargs.append(f"layer_type=\"{layer_type}\"")

        # zero points
        if hasattr(module, 'x_zero') and module.x_zero is not None:
            x_zero = get_var_name(target_name + "_x_zero")
            y_zero = get_var_name(target_name + "_y_zero")
            o_zero = get_var_name(target_name + "_o_zero")
            kwargs.append(f"x_zero={x_zero}")
            kwargs.append(f"y_zero={y_zero}")
            kwargs.append(f"o_zero={o_zero}")
        
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
            
        return f"{node.name} = dsl.qmatmul({params_str})"

    def build_QMatMulIsqrtD(self, node):
        """构建 QMatMulIsqrtD 的 DSL 调用 (带 sqrt(d) 归一化的 MatMul)"""
        # 与 QMatMul 基本相同，只是函数名不同
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        # 使用可能被边界转换替换的输入变量名
        inp1 = self._get_substituted_var_name(node, node.args[0], 0)
        inp2 = self._get_substituted_var_name(node, node.args[1], 1)
        
        x_scale_sign = get_var_name(target_name + "_x_scale_sign")
        x_scale_coe = get_var_name(target_name + "_x_scale_coe")
        x_scale_rshift = get_var_name(target_name + "_x_scale_rshift")
        
        y_scale_sign = get_var_name(target_name + "_y_scale_sign")
        y_scale_coe = get_var_name(target_name + "_y_scale_coe")
        y_scale_rshift = get_var_name(target_name + "_y_scale_rshift")

        fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
        fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
        fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
        
        o_scale_sign = get_var_name(target_name + "_o_scale_sign")
        o_scale_coe = get_var_name(target_name + "_o_scale_coe")
        o_scale_rshift = get_var_name(target_name + "_o_scale_rshift")
        
        o_scale_inv_sign = get_var_name(target_name + "_o_scale_inv_sign")
        o_scale_inv_coe = get_var_name(target_name + "_o_scale_inv_coe")
        o_scale_inv_rshift = get_var_name(target_name + "_o_scale_inv_rshift")
        
        params = [
            inp1, inp2,
            x_scale_sign, x_scale_coe, x_scale_rshift,
            y_scale_sign, y_scale_coe, y_scale_rshift,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            o_scale_sign, o_scale_coe, o_scale_rshift,
            o_scale_inv_sign, o_scale_inv_coe, o_scale_inv_rshift
        ]
        
        kwargs = []

        # layer_type: Q@K^T/sqrt(d)
        layer_type = "attn.QK"
        node_name_l = (node.name or "").lower()
        node_tgt_l = (getattr(node, "target", "") or "").lower()
        if "matmul2" in node_name_l or "matmul2" in node_tgt_l:
            # 兼容：若图里命名反了，尽量不标错
            layer_type = "attn.SV"
        kwargs.append(f"layer_type=\"{layer_type}\"")

        if hasattr(module, 'x_zero') and module.x_zero is not None:
            x_zero = get_var_name(target_name + "_x_zero")
            y_zero = get_var_name(target_name + "_y_zero")
            o_zero = get_var_name(target_name + "_o_zero")
            kwargs.append(f"x_zero={x_zero}")
            kwargs.append(f"y_zero={y_zero}")
            kwargs.append(f"o_zero={o_zero}")
        
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
        
        return f"{node.name} = dsl.qmatmul_isqrtd({params_str})"

    # ==================== Quant / Dequant 操作 ====================
    # 这些方法用于在量化/非量化边界处插入类型转换
    
    @staticmethod
    def generate_quant_call(output_name, input_name, quant_mode, 
                            scale_sign, scale_coe, scale_rshift, 
                            zero=None):
        """生成 quant DSL 调用代码
        
        Args:
            output_name: 输出变量名
            input_name: 输入变量名 (float)
            quant_mode: 量化模式 (0=sym_tensor, 1=asym_tensor, 2=sym_token, 3=asym_token)
            scale_sign, scale_coe, scale_rshift: scale 参数名
            zero: 零点参数名 (可选，仅 asymmetric 模式)
        
        Returns:
            DSL 代码字符串
        """
        params = [input_name, str(quant_mode), scale_sign, scale_coe, scale_rshift]
        if zero is not None:
            params.append(f"zero={zero}")
        return f"{output_name} = dsl.quant({', '.join(params)})"
    
    @staticmethod
    def generate_dequant_call(output_name, input_name, quant_mode,
                              scale_sign, scale_coe, scale_rshift,
                              zero=None):
        """生成 dequant DSL 调用代码
        
        Args:
            output_name: 输出变量名
            input_name: 输入变量名 (int)
            quant_mode: 量化模式 (0=sym_tensor, 1=asym_tensor, 2=sym_token, 3=asym_token)
            scale_sign, scale_coe, scale_rshift: scale 参数名
            zero: 零点参数名 (可选，仅 asymmetric 模式)
        
        Returns:
            DSL 代码字符串
        """
        params = [input_name, str(quant_mode), scale_sign, scale_coe, scale_rshift]
        if zero is not None:
            params.append(f"zero={zero}")
        return f"{output_name} = dsl.dequant({', '.join(params)})"