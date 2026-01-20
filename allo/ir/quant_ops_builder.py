"""Helper to build quantized ops from Python builder.

This module provides helper functions to emit Allo quantized operations
directly using the ops defined in AlloQuantOps.td.
"""
from .._mlir.ir import DenseI64ArrayAttr, InsertionPoint, Type
from .._mlir.dialects import allo as allo_d
from .utils import MockConstant


def build_quant_placeholder(ctx, node, attr, new_args, output_buffer, transformer_cls=None):
    """Build an Allo quantized operation directly using destination-passing style.

    Args:
        ctx: builder context
        node: AST node (unused, kept for compatibility)
        attr: operation name (string) - e.g., "qmatmul", "qconv2d", etc.
        new_args: list of built argument values (with .result and .type, or MockConstant for compile-time constants)
        output_buffer: pre-allocated memref for output (destination-passing style)
        transformer_cls: ASTTransformer class to visit keyword arguments

    Returns:
        The created Allo quantized operation (returns None in destination-passing style).
    
    Note:
        - Destination-passing style: output buffer is pre-allocated and passed as first operand
        - Operations write to the provided buffer and have no return values
        - For qconv2d, stride is passed as a DenseI64ArrayAttr attribute
        - Optional parameters (like bias, zeros) are passed as keyword arguments
    """
    # Convert arguments: materialize MockConstant to SSA values, keep existing Values
    arg_results = []
    for a in new_args:
        if isinstance(a, MockConstant):
            # Materialize the constant as an arith.constant op
            arg_results.append(a.result)
        elif hasattr(a, "result"):
            arg_results.append(a.result)
        else:
            # This should be a Value already
            arg_results.append(a)
    
    def get_kwarg(name, pos_idx=None, args_list=None):
        # Try kwarg first
        if hasattr(node, "keywords"):
            for kw in node.keywords:
                if kw.arg == name:
                    if transformer_cls:
                        val = transformer_cls()(ctx, kw.value)
                        # Unwrap MockConstant/Value
                        if isinstance(val, MockConstant):
                            return val.result
                        elif hasattr(val, "result"):
                            return val.result
                        return val
                    else:
                        # Fallback if no transformer
                        pass
        
        # Try positional
        if pos_idx is not None and args_list is not None and pos_idx < len(args_list):
            return args_list[pos_idx]
        
        return None

    # Get insertion point
    ip = ctx.get_ip()
    
    # Dispatch to the appropriate quantized operation
    # Note: Python bindings use destination-passing style - output buffer as first operand
    if attr == "qmatmul":
        # DSL call:
        # qmatmul(x, y,
        #        x_scale(sign,coe,rshift), y_scale(sign,coe,rshift),
        #        fused_scale(sign,coe,rshift),
        #        o_scale(sign,coe,rshift), o_scale_inv(sign,coe,rshift),
        #        x_zero?, y_zero?, o_zero?)
        # Python binding: QMatMulOp(output, lhs, rhs, x_scale..., y_scale..., fused_scale..., o_scale..., o_scale_inv..., x_zero=?, y_zero=?, o_zero=?)
        if len(arg_results) < 17:
            raise ValueError(
                f"qmatmul requires at least 17 operands (x, y, 5 scales x 3 components), got {len(arg_results)}"
            )
        lhs, rhs = arg_results[0], arg_results[1]
        # scales: each has (sign, coe, rshift) = 3 values
        x_scl_sign, x_scl_coe, x_scl_rshift = arg_results[2], arg_results[3], arg_results[4]
        y_scl_sign, y_scl_coe, y_scl_rshift = arg_results[5], arg_results[6], arg_results[7]
        fused_scl_sign, fused_scl_coe, fused_scl_rshift = arg_results[8], arg_results[9], arg_results[10]
        o_scl_sign, o_scl_coe, o_scl_rshift = arg_results[11], arg_results[12], arg_results[13]
        o_scl_inv_sign, o_scl_inv_coe, o_scl_inv_rshift = arg_results[14], arg_results[15], arg_results[16]
        
        x_zero = get_kwarg("x_zero", 17, arg_results)
        y_zero = get_kwarg("y_zero", 18, arg_results)
        o_zero = get_kwarg("o_zero", 19, arg_results)

        # Extract layer_type from keyword arguments or use default "unknown"
        layer_type = get_kwarg("layer_type", 20, arg_results)
        from .._mlir.ir import StringAttr
        layer_type_attr = StringAttr.get(layer_type if layer_type else "unknown")
        
        allo_d.QMatMulOp(output_buffer, lhs, rhs, x_scl_sign, x_scl_coe, x_scl_rshift, 
                         y_scl_sign, y_scl_coe, y_scl_rshift,
                         fused_scl_sign, fused_scl_coe, fused_scl_rshift,
                         o_scl_sign, o_scl_coe, o_scl_rshift,
                         o_scl_inv_sign, o_scl_inv_coe, o_scl_inv_rshift,
                         x_zero=x_zero, y_zero=y_zero, o_zero=o_zero,
                         layer_type=layer_type_attr, ip=ip)
        return output_buffer
    
    elif attr == "qmatmul_isqrtd":
        # DSL:
        # qmatmul_isqrtd(x, y,
        #              x_scale(sign,coe,rshift), y_scale(sign,coe,rshift),
        #              fused_scale(sign,coe,rshift),
        #              o_scale(sign,coe,rshift), o_scale_inv(sign,coe,rshift),
        #              x_zero?, y_zero?, o_zero?)
        # Binding: QMatMulIsqrtDOp(output, lhs, rhs, x_scale..., y_scale..., fused_scale..., o_scale..., o_scale_inv..., x_zero=?, y_zero=?, o_zero=?)
        if len(arg_results) < 17:
            raise ValueError(f"qmatmul_isqrtd requires at least 17 operands, got {len(arg_results)}")
        lhs, rhs = arg_results[0], arg_results[1]
        x_scl_sign, x_scl_coe, x_scl_rshift = arg_results[2], arg_results[3], arg_results[4]
        y_scl_sign, y_scl_coe, y_scl_rshift = arg_results[5], arg_results[6], arg_results[7]
        fused_scl_sign, fused_scl_coe, fused_scl_rshift = arg_results[8], arg_results[9], arg_results[10]
        o_scl_sign, o_scl_coe, o_scl_rshift = arg_results[11], arg_results[12], arg_results[13]
        o_scl_inv_sign, o_scl_inv_coe, o_scl_inv_rshift = arg_results[14], arg_results[15], arg_results[16]
        
        x_zero = get_kwarg("x_zero", 17, arg_results)
        y_zero = get_kwarg("y_zero", 18, arg_results)
        o_zero = get_kwarg("o_zero", 19, arg_results)

        # Extract layer_type from keyword arguments or use default "unknown"
        layer_type = get_kwarg("layer_type", 20, arg_results)
        from .._mlir.ir import StringAttr
        layer_type_attr = StringAttr.get(layer_type if layer_type else "unknown")
        
        allo_d.QMatMulIsqrtDOp(output_buffer, lhs, rhs, x_scl_sign, x_scl_coe, x_scl_rshift, 
                               y_scl_sign, y_scl_coe, y_scl_rshift,
                               fused_scl_sign, fused_scl_coe, fused_scl_rshift,
                               o_scl_sign, o_scl_coe, o_scl_rshift,
                               o_scl_inv_sign, o_scl_inv_coe, o_scl_inv_rshift,
                               x_zero=x_zero, y_zero=y_zero, o_zero=o_zero,
                               layer_type=layer_type_attr, ip=ip)
        return output_buffer
    
    elif attr == "qconv2d":
        # DSL call: qconv2d(input, filter, stride, fused_scl_sign, fused_scl_coe, fused_scl_rshift, input_scl..., output_scl..., weight_scl..., bias_scl..., bias=?)
        # Python binding: QConv2DOp(output, input, filter, fscl_sign, fscl_coe, fscl_rshift, iscl..., oscl..., wscl..., bscl..., stride, input_zero=?, output_zero=?, bias=?)
        if len(new_args) < 3:
            raise ValueError(f"qconv2d requires at least input, filter, stride, got {len(new_args)}")
        
        # Extract stride from new_args[2] (before conversion to SSA)
        stride_arg = new_args[2]
        stride = []
        if isinstance(stride_arg, list):
            for v in stride_arg:
                if isinstance(v, MockConstant):
                    stride.append(v.val)
                else:
                    stride.append(v)
        elif isinstance(stride_arg, MockConstant):
            stride_val = stride_arg.val
            stride = [stride_val, stride_val]
        else:
            stride = [1, 1]  # Default fallback
        
        stride_attr = DenseI64ArrayAttr.get(stride)
        
        # Rebuild arg_results without stride (index 2)
        filtered_results = [arg_results[i] for i in range(len(arg_results)) if i != 2]
        
        # Expected: input, filter, fused_scl(3), input_scl(3), output_scl(3), weight_scl(3) = 14 operands minimum
        if len(filtered_results) < 14:
            raise ValueError(f"qconv2d requires at least 14 operands (input, filter, 4 scales x 3 components), got {len(filtered_results)}")
        
        input_val, filter_val = filtered_results[0], filtered_results[1]
        fscl_sign, fscl_coe, fscl_rshift = filtered_results[2], filtered_results[3], filtered_results[4]
        iscl_sign, iscl_coe, iscl_rshift = filtered_results[5], filtered_results[6], filtered_results[7]
        oscl_sign, oscl_coe, oscl_rshift = filtered_results[8], filtered_results[9], filtered_results[10]
        oscl_inv_sign, oscl_inv_coe, oscl_inv_rshift = filtered_results[11], filtered_results[12], filtered_results[13]
        wscl_sign, wscl_coe, wscl_rshift = filtered_results[14], filtered_results[15], filtered_results[16]
        
        bscl_sign = get_kwarg("bscl_sign", 17, filtered_results)
        bscl_coe = get_kwarg("bscl_coe", 18, filtered_results)
        bscl_rshift = get_kwarg("bscl_rshift", 19, filtered_results)
        
        input_zero = get_kwarg("izr", 20, filtered_results)
        output_zero = get_kwarg("ozr", 21, filtered_results)
        bias = get_kwarg("bias", 22, filtered_results)
        
        allo_d.QConv2dOp(output_buffer, input_val, filter_val, 
                         fscl_sign, fscl_coe, fscl_rshift,
                         iscl_sign, iscl_coe, iscl_rshift,
                         oscl_sign, oscl_coe, oscl_rshift,
                         oscl_inv_sign, oscl_inv_coe, oscl_inv_rshift,
                         wscl_sign, wscl_coe, wscl_rshift,
                         bscl_sign=bscl_sign, bscl_coe=bscl_coe, bscl_rshift=bscl_rshift,
                         input_zero=input_zero, output_zero=output_zero, bias=bias, stride=stride_attr, ip=ip)
        return output_buffer
    
    elif attr == "qlinear":
        # DSL call: qlinear(input, weight, fused_scl_sign, fused_scl_coe, fused_scl_rshift, input_scl..., output_scl..., weight_scl..., bias_scl..., bias=?, layer_type=?)
        # Python binding: QLinearOp(output, input, weight, fscl..., iscl..., oscl..., wscl..., bscl..., input_zero=?, output_zero=?, bias=?, layer_type=?)
        if len(arg_results) < 14:
            raise ValueError(f"qlinear requires at least 14 operands (input, weight, 4 scales x 3 components), got {len(arg_results)}")
        
        input_val, weight_val = arg_results[0], arg_results[1]
        fscl_sign, fscl_coe, fscl_rshift = arg_results[2], arg_results[3], arg_results[4]
        iscl_sign, iscl_coe, iscl_rshift = arg_results[5], arg_results[6], arg_results[7]
        oscl_sign, oscl_coe, oscl_rshift = arg_results[8], arg_results[9], arg_results[10]
        oscl_inv_sign, oscl_inv_coe, oscl_inv_rshift = arg_results[11], arg_results[12], arg_results[13]
        wscl_sign, wscl_coe, wscl_rshift = arg_results[14], arg_results[15], arg_results[16]
        
        bscl_sign = get_kwarg("bscl_sign", 17, arg_results)
        bscl_coe = get_kwarg("bscl_coe", 18, arg_results)
        bscl_rshift = get_kwarg("bscl_rshift", 19, arg_results)
        
        input_zero = get_kwarg("izr", 20, arg_results)
        output_zero = get_kwarg("ozr", 21, arg_results)
        bias = get_kwarg("bias", 22, arg_results)
        
        # Extract layer_type from keyword arguments or use default "unknown"
        layer_type = get_kwarg("layer_type", 23, arg_results)
        
        # Build the QLinearOp with layer_type attribute
        from .._mlir.ir import StringAttr
        layer_type_attr = StringAttr.get(layer_type if layer_type else "unknown")
        
        allo_d.QLinearOp(output_buffer, input_val, weight_val,
                         fscl_sign, fscl_coe, fscl_rshift,
                         iscl_sign, iscl_coe, iscl_rshift,
                         oscl_sign, oscl_coe, oscl_rshift,
                         oscl_inv_sign, oscl_inv_coe, oscl_inv_rshift,
                         wscl_sign, wscl_coe, wscl_rshift,
                         bscl_sign=bscl_sign, bscl_coe=bscl_coe, bscl_rshift=bscl_rshift,
                         input_zero=input_zero, output_zero=output_zero, bias=bias,
                         layer_type=layer_type_attr, ip=ip)
        return output_buffer
    
    elif attr == "qadd":
        # DSL call: qadd(x, y, x_scale_sign, x_scale_coe, x_scale_rshift, y_scale..., o_scale...)
        # Python binding: QAddOp(output, lhs, rhs, x_scale..., y_scale..., o_scale..., x_zero=?, y_zero=?, o_zero=?)
        if len(arg_results) < 11:
            raise ValueError(f"qadd requires at least 11 operands (x, y, 3 scales x 3 components), got {len(arg_results)}")
        
        lhs, rhs = arg_results[0], arg_results[1]
        x_scl_sign, x_scl_coe, x_scl_rshift = arg_results[2], arg_results[3], arg_results[4]
        y_scl_sign, y_scl_coe, y_scl_rshift = arg_results[5], arg_results[6], arg_results[7]
        o_scl_sign, o_scl_coe, o_scl_rshift = arg_results[8], arg_results[9], arg_results[10]
        o_scl_inv_sign, o_scl_inv_coe, o_scl_inv_rshift = arg_results[11], arg_results[12], arg_results[13]
        
        x_zero = get_kwarg("x_zero", 14, arg_results)
        y_zero = get_kwarg("y_zero", 15, arg_results)
        o_zero = get_kwarg("o_zero", 16, arg_results)
        
        allo_d.QAddOp(output_buffer, lhs, rhs, 
                      x_scl_sign, x_scl_coe, x_scl_rshift, 
                      y_scl_sign, y_scl_coe, y_scl_rshift, 
                      o_scl_sign, o_scl_coe, o_scl_rshift,
                      o_scl_inv_sign, o_scl_inv_coe, o_scl_inv_rshift,
                      x_zero=x_zero, y_zero=y_zero, o_zero=o_zero, ip=ip)
        return output_buffer
    
    elif attr == "int_gelu":
        # DSL call: int_gelu(x, input_scale..., gelu_scale..., fused_scale..., output_scale..., input_zero?, output_zero?)
        # Python binding: IntGELUOp(output, input, input_scale..., gelu_scale..., fused_scale..., output_scale..., input_zero=?, output_zero=?)
        if len(arg_results) < 13:
            raise ValueError(f"int_gelu requires at least 13 operands (x, 4 scales x 3 components), got {len(arg_results)}")
        
        input_val = arg_results[0]
        in_scl_sign, in_scl_coe, in_scl_rshift = arg_results[1], arg_results[2], arg_results[3]
        gelu_scl_sign, gelu_scl_coe, gelu_scl_rshift = arg_results[4], arg_results[5], arg_results[6]
        fused_scl_sign, fused_scl_coe, fused_scl_rshift = arg_results[7], arg_results[8], arg_results[9]
        out_scl_sign, out_scl_coe, out_scl_rshift = arg_results[10], arg_results[11], arg_results[12]
        out_scl_inv_sign, out_scl_inv_coe, out_scl_inv_rshift = arg_results[13], arg_results[14], arg_results[15]
        
        input_zero = get_kwarg("input_zero", 16, arg_results)
        output_zero = get_kwarg("output_zero", 17, arg_results)
        
        allo_d.IntGELUOp(output_buffer, input_val, 
                         in_scl_sign, in_scl_coe, in_scl_rshift,
                         gelu_scl_sign, gelu_scl_coe, gelu_scl_rshift,
                         fused_scl_sign, fused_scl_coe, fused_scl_rshift,
                         out_scl_sign, out_scl_coe, out_scl_rshift,
                         out_scl_inv_sign, out_scl_inv_coe, out_scl_inv_rshift,
                         input_zero=input_zero, output_zero=output_zero, ip=ip)
        return output_buffer
    
    elif attr == "int_softmax":
        # DSL call: int_softmax(x, input_scale..., softmax_scale..., output_scale..., fused_scale..., input_zero?, output_zero?)
        # Python binding: IntSoftmaxOp(output, input, input_scale..., softmax_scale..., fused_scale..., output_scale..., input_zero=?, output_zero=?)
        if len(arg_results) < 10:
            raise ValueError(f"int_softmax requires at least 10 operands (x, 3 scales x 3 components), got {len(arg_results)}")
        
        input_val = arg_results[0]
        in_scl_sign, in_scl_coe, in_scl_rshift = arg_results[1], arg_results[2], arg_results[3]
        soft_scl_sign, soft_scl_coe, soft_scl_rshift = arg_results[4], arg_results[5], arg_results[6]
        # Note: dsl.py has output_scale BEFORE fused_scale
        out_scl_sign, out_scl_coe, out_scl_rshift = arg_results[7], arg_results[8], arg_results[9]
        out_scl_inv_sign, out_scl_inv_coe, out_scl_inv_rshift = arg_results[10], arg_results[11], arg_results[12]
        
        fused_scl_sign = get_kwarg("fused_scale_sign", 13, arg_results)
        fused_scl_coe = get_kwarg("fused_scale_coe", 14, arg_results)
        fused_scl_rshift = get_kwarg("fused_scale_rshift", 15, arg_results)
        
        input_zero = get_kwarg("input_zero", 16, arg_results)
        output_zero = get_kwarg("output_zero", 17, arg_results)
        
        allo_d.IntSoftmaxOp(output_buffer, input_val, 
                            in_scl_sign, in_scl_coe, in_scl_rshift,
                            soft_scl_sign, soft_scl_coe, soft_scl_rshift,
                            out_scl_sign, out_scl_coe, out_scl_rshift,
                            out_scl_inv_sign, out_scl_inv_coe, out_scl_inv_rshift,
                            fused_scale_sign=fused_scl_sign, fused_scale_coe=fused_scl_coe, fused_scale_rshift=fused_scl_rshift,
                            input_zero=input_zero, output_zero=output_zero, ip=ip)
        return output_buffer
    
    elif attr == "int_layernorm":
        # DSL call: int_layernorm(x, bias_int, input_scale..., layernorm_scale..., bias_scale..., fused_scale..., output_scale...)
        # Python binding: IntLayerNormOp(output, input, bias_int, input_scale..., layernorm_scale..., bias_scale..., fused_scale..., output_scale..., input_zero=?, output_zero=?)
        if len(arg_results) < 17:
            raise ValueError(f"int_layernorm requires at least 17 operands (x, bias, 5 scales x 3 components), got {len(arg_results)}")
        
        input_val, bias_int = arg_results[0], arg_results[1]
        in_scl_sign, in_scl_coe, in_scl_rshift = arg_results[2], arg_results[3], arg_results[4]
        ln_scl_sign, ln_scl_coe, ln_scl_rshift = arg_results[5], arg_results[6], arg_results[7]
        bias_scl_sign, bias_scl_coe, bias_scl_rshift = arg_results[8], arg_results[9], arg_results[10]
        fused_scl_sign, fused_scl_coe, fused_scl_rshift = arg_results[11], arg_results[12], arg_results[13]
        out_scl_sign, out_scl_coe, out_scl_rshift = arg_results[14], arg_results[15], arg_results[16]
        out_scl_inv_sign, out_scl_inv_coe, out_scl_inv_rshift = arg_results[17], arg_results[18], arg_results[19]
        
        input_zero = get_kwarg("input_zero", 20, arg_results)
        output_zero = get_kwarg("output_zero", 21, arg_results)
        
        allo_d.IntLayerNormOp(output_buffer, input_val, bias_int,
                              in_scl_sign, in_scl_coe, in_scl_rshift,
                              ln_scl_sign, ln_scl_coe, ln_scl_rshift,
                              bias_scl_sign, bias_scl_coe, bias_scl_rshift,
                              fused_scl_sign, fused_scl_coe, fused_scl_rshift,
                              out_scl_sign, out_scl_coe, out_scl_rshift,
                              out_scl_inv_sign, out_scl_inv_coe, out_scl_inv_rshift,
                              input_zero=input_zero, output_zero=output_zero, ip=ip)
        return output_buffer
    
    elif attr == "quant":
        # DSL call: quant(x, quant_mode, scale_sign, scale_coe, scale_rshift, zero=?)
        # Python binding: QuantOp(output, input, scale_sign, scale_coe, scale_rshift, zero=?, quant_mode=N)
        # x: float input, output: int
        if len(arg_results) < 5:
            raise ValueError(f"quant requires at least 5 operands (x, quant_mode, scale_sign, scale_coe, scale_rshift), got {len(arg_results)}")
        
        input_val = arg_results[0]
        # quant_mode is passed as second positional arg (index 1)
        # Extract it as integer value for attribute
        quant_mode_arg = new_args[1]  # Use new_args to get the original MockConstant
        if isinstance(quant_mode_arg, MockConstant):
            quant_mode = int(quant_mode_arg.val)
        else:
            quant_mode = 0  # Default: symmetric, per-tensor
        
        scl_sign, scl_coe, scl_rshift = arg_results[2], arg_results[3], arg_results[4]
        zero = get_kwarg("zero", 5, arg_results)
        
        from .._mlir.ir import IntegerAttr, IntegerType
        quant_mode_attr = IntegerAttr.get(IntegerType.get_signless(8), quant_mode)
        
        allo_d.QuantOp(output_buffer, input_val,
                       scl_sign, scl_coe, scl_rshift,
                       zero=zero, quant_mode=quant_mode_attr, ip=ip)
        return output_buffer
    
    elif attr == "dequant":
        # DSL call: dequant(x, quant_mode, scale_sign, scale_coe, scale_rshift, zero=?)
        # Python binding: DequantOp(output, input, scale_sign, scale_coe, scale_rshift, zero=?, quant_mode=N)
        # x: int input, output: float
        if len(arg_results) < 5:
            raise ValueError(f"dequant requires at least 5 operands (x, quant_mode, scale_sign, scale_coe, scale_rshift), got {len(arg_results)}")
        
        input_val = arg_results[0]
        # quant_mode is passed as second positional arg (index 1)
        quant_mode_arg = new_args[1]  # Use new_args to get the original MockConstant
        if isinstance(quant_mode_arg, MockConstant):
            quant_mode = int(quant_mode_arg.val)
        else:
            quant_mode = 0  # Default: symmetric, per-tensor
        
        scl_sign, scl_coe, scl_rshift = arg_results[2], arg_results[3], arg_results[4]
        zero = get_kwarg("zero", 5, arg_results)
        
        from .._mlir.ir import IntegerAttr, IntegerType
        quant_mode_attr = IntegerAttr.get(IntegerType.get_signless(8), quant_mode)
        
        allo_d.DequantOp(output_buffer, input_val,
                         scl_sign, scl_coe, scl_rshift,
                         zero=zero, quant_mode=quant_mode_attr, ip=ip)
        return output_buffer
    
    else:
        raise ValueError(f"Unknown quantized operation: {attr}")
