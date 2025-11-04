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
from .library import CoreAttention_lib, KVCache_lib, ViTGetFirstToken_lib, ViTTokenExpand_lib
from ..quant.quant_modules import *
from .. import dsl
from ..ir import types
from ..customize import customize


def _process_quantized_params(gm, global_vars):
    """
    处理量化模块的参数和buffers，将浮点权重/偏置替换为整数版本，
    并注入所有量化相关的scale/zero常量到global_vars中。
    
    Scale参数会被转换为定点表示：scale = sign * coe * 2^(-rshift)
    其中 coe ∈ [0.5, 1)，存储为定点数（使用 uint16/uint32）
    
    Args:
        gm: GraphModule，包含traced的模型
        global_vars: 全局变量字典，将被就地修改
    
    Returns:
        None (就地修改global_vars)
    """
    import numpy as np
    
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
            coe_fixed: 定点系数，int16，范围 [2^15, 2^16)，对应浮点 [0.5, 1)
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
        max_iterations = 10  # 防止无限循环
        
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
        
        # coe_float ∈ [0.5, 1) → coe_fixed ∈ [2^15, 2^16)
        # 注意：使用 uint16 表示更合理，但为了兼容性暂时用 int16 返回
        max_fixed_value = (1 << fixed_bits) - 1  # 2^16 - 1 = 65535
        min_fixed_value = (1 << (fixed_bits - 1))  # 2^15 = 32768
        
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
        
        # 此处均以i32返回
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
                    sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
                    
                    global_vars[var_prefix + "_weight_scale_sign"] = sign.astype(np.int8)
                    global_vars[var_prefix + "_weight_scale_coe"] = coe.astype(np.uint16)
                    global_vars[var_prefix + "_weight_scale_rshift"] = rshift.astype(np.int16)
        
        # ========== 处理偏置 (bias) ==========
        if hasattr(module, 'bias') and module.bias is not None:
            if hasattr(module, 'bias_int'):
                bias_int_data = module.bias_int.data.detach().numpy()
                bias_key = var_prefix + "_bias_int"
                global_vars[bias_key] = bias_int_data.astype(np.int32)
                
                # 注入bias_scale（定点表示）
                if hasattr(module, 'bias_scale'):
                    scale_data = module.bias_scale.data.detach().numpy()
                    sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
                    
                    global_vars[var_prefix + "_bias_scale_sign"] = sign.astype(np.int8)
                    global_vars[var_prefix + "_bias_scale_coe"] = coe.astype(np.uint16)
                    global_vars[var_prefix + "_bias_scale_rshift"] = rshift.astype(np.int16)
        
        # ========== 处理激活量化相关的scale（定点表示）==========
        
        # input_scale
        if hasattr(module, 'input_scale'):
            scale_data = module.input_scale.data.detach().numpy()
            sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
            
            global_vars[var_prefix + "_input_scale_sign"] = sign.astype(np.int8)
            global_vars[var_prefix + "_input_scale_coe"] = coe.astype(np.uint16)
            global_vars[var_prefix + "_input_scale_rshift"] = rshift.astype(np.int16)
        
        # output_scale
        if hasattr(module, 'output_scale'):
            scale_data = module.output_scale.data.detach().numpy()
            sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
            
            global_vars[var_prefix + "_output_scale_sign"] = sign.astype(np.int8)
            global_vars[var_prefix + "_output_scale_coe"] = coe.astype(np.uint16)
            global_vars[var_prefix + "_output_scale_rshift"] = rshift.astype(np.int16)
        
        # fused_scale（最关键：用于运行时缩放）
        if hasattr(module, 'fused_scale'):
            scale_data = module.fused_scale.data.detach().numpy()
            sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
            
            global_vars[var_prefix + "_fused_scale_sign"] = sign.astype(np.int8)
            global_vars[var_prefix + "_fused_scale_coe"] = coe.astype(np.uint16)
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
                sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
                
                global_vars[var_prefix + "_layernorm_scale_sign"] = sign.astype(np.int8)
                global_vars[var_prefix + "_layernorm_scale_coe"] = coe.astype(np.uint16)
                global_vars[var_prefix + "_layernorm_scale_rshift"] = rshift.astype(np.int16)
            
            if hasattr(module, 'bias_int'):
                bias_int_data = module.bias_int.data.detach().numpy()
                global_vars[var_prefix + "_bias_int"] = bias_int_data.astype(np.int32)
        
        # ========== 特殊处理：QAdd/QMatMul的双输入scale ==========
        if isinstance(module, (QAdd, QMatMul, QMatMulIsqrtD)):
            # x_scale, y_scale, o_scale
            for scale_name in ['x_scale', 'y_scale', 'o_scale']:
                if hasattr(module, scale_name):
                    scale_data = getattr(module, scale_name).data.detach().numpy()
                    sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
                    
                    global_vars[var_prefix + f"_{scale_name}_sign"] = sign.astype(np.int8)
                    global_vars[var_prefix + f"_{scale_name}_coe"] = coe.astype(np.uint16)
                    global_vars[var_prefix + f"_{scale_name}_rshift"] = rshift.astype(np.int16)
            
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
                sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
                
                global_vars[var_prefix + "_softmax_scale_sign"] = sign.astype(np.int8)
                global_vars[var_prefix + "_softmax_scale_coe"] = coe.astype(np.uint16)
                global_vars[var_prefix + "_softmax_scale_rshift"] = rshift.astype(np.int16)
        
        # ========== 特殊处理：IntGELU ==========
        if isinstance(module, IntGELU):
            if hasattr(module, 'gelu_scale'):
                scale_data = module.gelu_scale.data.detach().numpy()
                sign, coe, rshift = float_to_fixed_point(scale_data, fixed_bits=16)
                
                global_vars[var_prefix + "_gelu_scale_sign"] = sign.astype(np.int8)
                global_vars[var_prefix + "_gelu_scale_coe"] = coe.astype(np.uint16)
                global_vars[var_prefix + "_gelu_scale_rshift"] = rshift.astype(np.int16)

def from_pytorch(
    model,
    example_inputs,
    leaf_modules=None,
    verbose=False,
    enable_tensor=False,
    target="llvm",
    mode="csim",
    project="top.prj",
):
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
    # 处理参数 (nn.Parameter)
    for name, param in gm.named_parameters():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: param.detach().numpy()})
    # 处理 buffers (register_buffer)
    # 非FPGA必需量化的话，反而不需要_preprocess_quantized_params
    for name, buffer in gm.named_buffers():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: buffer.detach().numpy()})

    builder = TorchBuilder(gm, example_inputs, leaf_modules)
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
    def __init__(self, gm, example_inputs, leaf_modules=None):
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
        args = [
            (
                f"{name}: float32[{', '.join([str(s) for s in shape])}]"
                if shape
                else f"{name}: int32"
            )
            for name, shape in zip(self.input_args, self.input_shapes)
        ]
        res = ""
        # top-level function
        res += f"def forward({', '.join(args)})".format()
        # outputs
        res += f" -> ({', '.join(self.output)}):\n"
        # subfunctions
        if self.subfunctions:
            res += "\n".join(self.subfunctions) + "\n"
        if self.named_params:
            for name, param in self.named_params:
                new_name = name.replace(".", "_")
                res += f"    {new_name}: float32[{', '.join([str(s) for s in param.shape])}] = g_{new_name}\n"
        # if self.named_buffers:
        #     for name, buffer in self.named_buffers:
        #         new_name = name.replace(".", "_")
        #         # 根据 buffer 的实际 dtype 生成类型标注
        #         dtype_str = str(buffer.dtype).replace("torch.", "")
        #         if buffer.ndim == 0:
        #             # 标量 buffer
        #             res += f"    {new_name}: {dtype_str} = g_{new_name}\n"
        #         else:
        #             # 张量 buffer
        #             res += f"    {new_name}: {dtype_str}[{', '.join([str(s) for s in buffer.shape])}] = g_{new_name}\n"
    
        # ========== 新增：处理量化模块的定点参数 ==========
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
        
        # 遍历所有命名模块
        for module_name, module in self.gm.named_modules():
            if not isinstance(module, quantized_module_types):
                continue
            
            # 模块名转换为变量名格式
            var_prefix = module_name.replace(".", "_")
            
            # ========== 权重相关 ==========
            if hasattr(module, 'weight_int'):
                # weight_int: int8[shape]
                if hasattr(module.weight_int, 'shape'):
                    shape = module.weight_int.shape
                    shape_str = ', '.join(str(s) for s in shape)
                    declarations.append(f"    {var_prefix}_weight_int: int8[{shape_str}] = g_{var_prefix}_weight_int")
                
                # weight_scale 的定点表示（sign, coe, rshift）
                if hasattr(module, 'weight_scale'):
                    scale_shape = module.weight_scale.shape
                    if len(scale_shape) == 0:
                        # 标量
                        # !!!: Please Notice here if you need to change the type (int8 here) of weight_scale_sign to other type
                        declarations.append(f"    {var_prefix}_weight_scale_sign: int8 = g_{var_prefix}_weight_scale_sign")
                        declarations.append(f"    {var_prefix}_weight_scale_coe: uint16 = g_{var_prefix}_weight_scale_coe")
                        declarations.append(f"    {var_prefix}_weight_scale_rshift: int16 = g_{var_prefix}_weight_scale_rshift")
                    else:
                        # 向量/张量
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_weight_scale_sign: int8[{shape_str}] = g_{var_prefix}_weight_scale_sign")
                        declarations.append(f"    {var_prefix}_weight_scale_coe: uint16[{shape_str}] = g_{var_prefix}_weight_scale_coe")
                        declarations.append(f"    {var_prefix}_weight_scale_rshift: int16[{shape_str}] = g_{var_prefix}_weight_scale_rshift")
            
            # ========== 偏置相关 ==========
            if hasattr(module, 'bias_int'):
                if hasattr(module.bias_int, 'shape'):
                    bias_shape = module.bias_int.shape
                    bias_shape_str = ', '.join(str(s) for s in bias_shape)
                    declarations.append(f"    {var_prefix}_bias_int: int32[{bias_shape_str}] = g_{var_prefix}_bias_int")
                
                if hasattr(module, 'bias_scale'):
                    scale_shape = module.bias_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_bias_scale_sign: int8 = g_{var_prefix}_bias_scale_sign")
                        declarations.append(f"    {var_prefix}_bias_scale_coe: uint16 = g_{var_prefix}_bias_scale_coe")
                        declarations.append(f"    {var_prefix}_bias_scale_rshift: int16 = g_{var_prefix}_bias_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_bias_scale_sign: int8[{shape_str}] = g_{var_prefix}_bias_scale_sign")
                        declarations.append(f"    {var_prefix}_bias_scale_coe: uint16[{shape_str}] = g_{var_prefix}_bias_scale_coe")
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
                            declarations.append(f"    {var_prefix}_{scale_name}_coe: uint16 = g_{var_prefix}_{scale_name}_coe")
                            declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16 = g_{var_prefix}_{scale_name}_rshift")
                        else:
                            # 向量/张量
                            shape_str = ', '.join(str(s) for s in scale_shape)
                            declarations.append(f"    {var_prefix}_{scale_name}_sign: int8[{shape_str}] = g_{var_prefix}_{scale_name}_sign")
                            declarations.append(f"    {var_prefix}_{scale_name}_coe: uint16[{shape_str}] = g_{var_prefix}_{scale_name}_coe")
                            declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16[{shape_str}] = g_{var_prefix}_{scale_name}_rshift")
            
            # ========== Zero points（非对称量化）==========
            zero_names = ['input_zero', 'output_zero']
            for zero_name in zero_names:
                if hasattr(module, zero_name):
                    zero_attr = getattr(module, zero_name)
                    if zero_attr is not None and hasattr(zero_attr, 'shape'):
                        zero_shape = zero_attr.shape
                        if len(zero_shape) == 0:
                            declarations.append(f"    {var_prefix}_{zero_name}: int8 = g_{var_prefix}_{zero_name}")
                        else:
                            shape_str = ', '.join(str(s) for s in zero_shape)
                            declarations.append(f"    {var_prefix}_{zero_name}: int8[{shape_str}] = g_{var_prefix}_{zero_name}")
            
            # ========== 特殊处理：IntLayerNorm ==========
            if isinstance(module, IntLayerNorm):
                if hasattr(module, 'layernorm_scale'):
                    scale_shape = module.layernorm_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_layernorm_scale_sign: int8 = g_{var_prefix}_layernorm_scale_sign")
                        declarations.append(f"    {var_prefix}_layernorm_scale_coe: uint16 = g_{var_prefix}_layernorm_scale_coe")
                        declarations.append(f"    {var_prefix}_layernorm_scale_rshift: int16 = g_{var_prefix}_layernorm_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_layernorm_scale_sign: int8[{shape_str}] = g_{var_prefix}_layernorm_scale_sign")
                        declarations.append(f"    {var_prefix}_layernorm_scale_coe: uint16[{shape_str}] = g_{var_prefix}_layernorm_scale_coe")
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
                                declarations.append(f"    {var_prefix}_{scale_name}_coe: uint16 = g_{var_prefix}_{scale_name}_coe")
                                declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16 = g_{var_prefix}_{scale_name}_rshift")
                            else:
                                shape_str = ', '.join(str(s) for s in scale_shape)
                                declarations.append(f"    {var_prefix}_{scale_name}_sign: int8[{shape_str}] = g_{var_prefix}_{scale_name}_sign")
                                declarations.append(f"    {var_prefix}_{scale_name}_coe: uint16[{shape_str}] = g_{var_prefix}_{scale_name}_coe")
                                declarations.append(f"    {var_prefix}_{scale_name}_rshift: int16[{shape_str}] = g_{var_prefix}_{scale_name}_rshift")
                
                for zero_name in ['x_zero', 'y_zero', 'o_zero']:
                    if hasattr(module, zero_name):
                        zero_attr = getattr(module, zero_name)
                        if zero_attr is not None and hasattr(zero_attr, 'shape'):
                            zero_shape = zero_attr.shape
                            if len(zero_shape) == 0:
                                declarations.append(f"    {var_prefix}_{zero_name}: int8 = g_{var_prefix}_{zero_name}")
                            else:
                                shape_str = ', '.join(str(s) for s in zero_shape)
                                declarations.append(f"    {var_prefix}_{zero_name}: int8[{shape_str}] = g_{var_prefix}_{zero_name}")
            
            # ========== 特殊处理：IntSoftmax ==========
            if isinstance(module, IntSoftmax):
                if hasattr(module, 'softmax_scale'):
                    scale_shape = module.softmax_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_softmax_scale_sign: int8 = g_{var_prefix}_softmax_scale_sign")
                        declarations.append(f"    {var_prefix}_softmax_scale_coe: uint16 = g_{var_prefix}_softmax_scale_coe")
                        declarations.append(f"    {var_prefix}_softmax_scale_rshift: int16 = g_{var_prefix}_softmax_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_softmax_scale_sign: int8[{shape_str}] = g_{var_prefix}_softmax_scale_sign")
                        declarations.append(f"    {var_prefix}_softmax_scale_coe: uint16[{shape_str}] = g_{var_prefix}_softmax_scale_coe")
                        declarations.append(f"    {var_prefix}_softmax_scale_rshift: int16[{shape_str}] = g_{var_prefix}_softmax_scale_rshift")
            
            # ========== 特殊处理：IntGELU ==========
            if isinstance(module, IntGELU):
                if hasattr(module, 'gelu_scale'):
                    scale_shape = module.gelu_scale.shape
                    if len(scale_shape) == 0:
                        declarations.append(f"    {var_prefix}_gelu_scale_sign: int8 = g_{var_prefix}_gelu_scale_sign")
                        declarations.append(f"    {var_prefix}_gelu_scale_coe: uint16 = g_{var_prefix}_gelu_scale_coe")
                        declarations.append(f"    {var_prefix}_gelu_scale_rshift: int16 = g_{var_prefix}_gelu_scale_rshift")
                    else:
                        shape_str = ', '.join(str(s) for s in scale_shape)
                        declarations.append(f"    {var_prefix}_gelu_scale_sign: int8[{shape_str}] = g_{var_prefix}_gelu_scale_sign")
                        declarations.append(f"    {var_prefix}_gelu_scale_coe: uint16[{shape_str}] = g_{var_prefix}_gelu_scale_coe")
                        declarations.append(f"    {var_prefix}_gelu_scale_rshift: int16[{shape_str}] = g_{var_prefix}_gelu_scale_rshift")
        
        # 返回所有声明，每个声明一行
        return '\n'.join(declarations) + '\n' if declarations else ''

    def __call__(self, node):
        method = getattr(self, "build_" + node.op)
        ret = method(node)
        if ret:
            self.code.append(ret)
        return ret

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
                        return getattr(self, f"build_{module.__class__.__name__}")(node)
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

    def append_output(self, output):
        shape = str(list(output.shape))
        dtype = str(output.dtype)[6:]
        self.output.append(dtype + shape)

    def build_output(self, node):
        if isinstance(node.meta["tensor_meta"], TensorMetadata):
            self.append_output(node.meta["tensor_meta"])
        elif isinstance(node.meta["tensor_meta"], (list, tuple)):
            for output in node.meta["tensor_meta"]:
                if isinstance(output, TensorMetadata):
                    self.append_output(output)
                elif isinstance(output, (list, tuple)):
                    for item in output:
                        if isinstance(item, TensorMetadata):
                            self.append_output(item)
                elif isinstance(output, dict):
                    for item in output.values():
                        if isinstance(item, TensorMetadata):
                            self.append_output(item)
                        else:
                            raise NotImplementedError("Unsupported output type")
        elif isinstance(node.meta["tensor_meta"], dict):
            for output in node.meta["tensor_meta"].values():
                if isinstance(output, TensorMetadata):
                    self.append_output(output)
        # Unwrap all outputs and return them
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
    
    def build_ViTGetFirstToken(self, node, shape):
        shape = (self.example_inputs[0].shape[0], shape[1], shape[2])
        src = inspect.getsource(ViTGetFirstToken_lib(*shape))
        src = (
            src.replace("s_0", str(shape[0]))
            .replace("s_1", str(shape[1]))
            .replace("s_2", str(shape[2]))
        )
        if src not in self.subfunctions:
            self.subfunctions.append(src)
        return f"{node.name} = ViTGetFirstToken({', '.join([get_var_name(arg) for arg in node.args])})"
    
    def build_ViTTokenExpand(self, node, shape):
        shape = (self.example_inputs[0].shape[0], shape[1], shape[2])
        src = inspect.getsource(ViTTokenExpand_lib(*shape))
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
        inp = get_var_name(node.args[0])
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
        
        # weight_scale 的拆分参数
        weight_scale_sign = get_var_name(target_name + "_weight_scale_sign")
        weight_scale_coe = get_var_name(target_name + "_weight_scale_coe")
        weight_scale_rshift = get_var_name(target_name + "_weight_scale_rshift")
        
        # bias_scale 的拆分参数
        bias_scale_sign = get_var_name(target_name + "_bias_scale_sign")
        bias_scale_coe = get_var_name(target_name + "_bias_scale_coe")
        bias_scale_rshift = get_var_name(target_name + "_bias_scale_rshift")
        
        params = [
            inp, weight, stride,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            input_scale_sign, input_scale_coe, input_scale_rshift,
            output_scale_sign, output_scale_coe, output_scale_rshift,
            weight_scale_sign, weight_scale_coe, weight_scale_rshift,
            bias_scale_sign, bias_scale_coe, bias_scale_rshift
        ]
        
        kwargs = []
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            kwargs.append(f"izr={input_zero}")
        
        if hasattr(module, 'output_zero') and module.output_zero is not None:
            output_zero = get_var_name(target_name + "_output_zero")
            kwargs.append(f"ozr={output_zero}")
        
        if module.bias is not None:
            bias = get_var_name(target_name + "_bias_int")
            kwargs.append(f"bias={bias}")
        
        # 组装参数字符串
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
        
        return f"{node.name} = dsl.qconv2d({params_str})"

    def build_QLinear(self, node):
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
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
        
        # weight_scale 的拆分参数
        weight_scale_sign = get_var_name(target_name + "_weight_scale_sign")
        weight_scale_coe = get_var_name(target_name + "_weight_scale_coe")
        weight_scale_rshift = get_var_name(target_name + "_weight_scale_rshift")
        
        # bias_scale 的拆分参数
        bias_scale_sign = get_var_name(target_name + "_bias_scale_sign")
        bias_scale_coe = get_var_name(target_name + "_bias_scale_coe")
        bias_scale_rshift = get_var_name(target_name + "_bias_scale_rshift")
        
        params = [
            inp, weight,
            fused_scale_sign, fused_scale_coe, fused_scale_rshift,
            input_scale_sign, input_scale_coe, input_scale_rshift,
            output_scale_sign, output_scale_coe, output_scale_rshift,
            weight_scale_sign, weight_scale_coe, weight_scale_rshift,
            bias_scale_sign, bias_scale_coe, bias_scale_rshift
        ]
        
        kwargs = []
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            kwargs.append(f"izr={input_zero}")
        
        if hasattr(module, 'output_zero') and module.output_zero is not None:
            output_zero = get_var_name(target_name + "_output_zero")
            kwargs.append(f"ozr={output_zero}")
        
        if module.bias is not None:
            bias = get_var_name(target_name + "_bias_int")
            kwargs.append(f"bias={bias}")
        
        # 组装参数字符串
        params_str = ', '.join(str(p) for p in params)
        if kwargs:
            params_str += ', ' + ', '.join(kwargs)
        
        return f"{node.name} = dsl.qlinear({params_str})"
    
    def build_QAdd(self, node):
        """构建 QAdd 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp1 = get_var_name(node.args[0])
        inp2 = get_var_name(node.args[1])
        
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
        
        # zero points
        if hasattr(module, 'x_zero') and module.x_zero is not None:
            x_zero = get_var_name(target_name + "_x_zero")
            y_zero = get_var_name(target_name + "_y_zero")
            o_zero = get_var_name(target_name + "_o_zero")
    
        return f"{node.name} = dsl.qadd({inp1}, {inp2}, {x_scale_sign}, {x_scale_coe}, {x_scale_rshift}, {y_scale_sign}, {y_scale_coe}, {y_scale_rshift}, {o_scale_sign}, {o_scale_coe}, {o_scale_rshift})"
    
    def build_IntSoftmax(self, node):
        """构建 IntSoftmax 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        
        # 获取 scale 参数
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
        softmax_scale_sign = get_var_name(target_name + "_softmax_scale_sign")
        softmax_scale_coe = get_var_name(target_name + "_softmax_scale_coe")
        softmax_scale_rshift = get_var_name(target_name + "_softmax_scale_rshift")
        
        # 如果有 fused_scale
        if hasattr(module, 'fused_scale'):
            fused_scale_sign = get_var_name(target_name + "_fused_scale_sign")
            fused_scale_coe = get_var_name(target_name + "_fused_scale_coe")
            fused_scale_rshift = get_var_name(target_name + "_fused_scale_rshift")
        
        # zero points (如果是非对称量化)
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            output_zero = get_var_name(target_name + "_output_zero")
        
        # 构建 DSL 调用
        return f"{node.name} = dsl.int_softmax({inp}, {input_scale_sign}, {input_scale_coe}, {input_scale_rshift}, {softmax_scale_sign}, {softmax_scale_coe}, {softmax_scale_rshift}, {output_scale_sign}, {output_scale_coe}, {output_scale_rshift})"
    
    def build_IntLayerNorm(self, node):
        """构建 IntLayerNorm 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        
        # bias_int
        bias_int = get_var_name(target_name + "_bias_int")
        
        # 获取 scale 参数
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
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
        
        # zero points
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            output_zero = get_var_name(target_name + "_output_zero")
        
        return f"{node.name} = dsl.int_layernorm({inp}, {bias_int}, {layernorm_scale_sign}, {layernorm_scale_coe}, {layernorm_scale_rshift}, {bias_scale_sign}, {bias_scale_coe}, {bias_scale_rshift}, {fused_scale_sign}, {fused_scale_coe}, {fused_scale_rshift})"
    
    def build_IntGELU(self, node):
        """构建 IntGELU 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        
        # 获取 scale 参数
        input_scale_sign = get_var_name(target_name + "_input_scale_sign")
        input_scale_coe = get_var_name(target_name + "_input_scale_coe")
        input_scale_rshift = get_var_name(target_name + "_input_scale_rshift")
        
        output_scale_sign = get_var_name(target_name + "_output_scale_sign")
        output_scale_coe = get_var_name(target_name + "_output_scale_coe")
        output_scale_rshift = get_var_name(target_name + "_output_scale_rshift")
        
        gelu_scale_sign = get_var_name(target_name + "_gelu_scale_sign")
        gelu_scale_coe = get_var_name(target_name + "_gelu_scale_coe")
        gelu_scale_rshift = get_var_name(target_name + "_gelu_scale_rshift")
        
        # zero points
        if hasattr(module, 'input_zero') and module.input_zero is not None:
            input_zero = get_var_name(target_name + "_input_zero")
            output_zero = get_var_name(target_name + "_output_zero")
        
        return f"{node.name} = dsl.int_gelu({inp}, {gelu_scale_sign}, {gelu_scale_coe}, {gelu_scale_rshift}, {output_scale_sign}, {output_scale_coe}, {output_scale_rshift})"
    def build_QMatMul(self, node):
        """构建 QMatMul 的 DSL 调用"""
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp1 = get_var_name(node.args[0])
        inp2 = get_var_name(node.args[1])
        
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
        
        # zero points
        if hasattr(module, 'x_zero') and module.x_zero is not None:
            x_zero = get_var_name(target_name + "_x_zero")
            y_zero = get_var_name(target_name + "_y_zero")
            o_zero = get_var_name(target_name + "_o_zero")
        
        return f"{node.name} = dsl.qmatmul({inp1}, {inp2}, {x_scale_sign}, {x_scale_coe}, {x_scale_rshift}, {y_scale_sign}, {y_scale_coe}, {y_scale_rshift}, {o_scale_sign}, {o_scale_coe}, {o_scale_rshift})"

    def build_QMatMulIsqrtD(self, node):
        """构建 QMatMulIsqrtD 的 DSL 调用 (带 sqrt(d) 归一化的 MatMul)"""
        # 与 QMatMul 基本相同，只是函数名不同
        module = self.get_module(node.target)
        target_name = node.target.replace(".", "_")
        inp1 = get_var_name(node.args[0])
        inp2 = get_var_name(node.args[1])
        
        x_scale_sign = get_var_name(target_name + "_x_scale_sign")
        x_scale_coe = get_var_name(target_name + "_x_scale_coe")
        x_scale_rshift = get_var_name(target_name + "_x_scale_rshift")
        
        y_scale_sign = get_var_name(target_name + "_y_scale_sign")
        y_scale_coe = get_var_name(target_name + "_y_scale_coe")
        y_scale_rshift = get_var_name(target_name + "_y_scale_rshift")
        
        o_scale_sign = get_var_name(target_name + "_o_scale_sign")
        o_scale_coe = get_var_name(target_name + "_o_scale_coe")
        o_scale_rshift = get_var_name(target_name + "_o_scale_rshift")
        
        return f"{node.name} = dsl.qmatmul_isqrtd({inp1}, {inp2}, {x_scale_sign}, {x_scale_coe}, {x_scale_rshift}, {y_scale_sign}, {y_scale_coe}, {y_scale_rshift}, {o_scale_sign}, {o_scale_coe}, {o_scale_rshift})"