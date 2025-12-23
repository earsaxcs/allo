/*
 * Allo Quantized Ops - Precise type definitions based on frontend usage
 * These ops accept both float and integer tensor/memref inputs/outputs.
 * Scale parameters are passed as integer arrays (sign: i8, coe: i16, rshift: i16).
 * Optional zero points and bias are handled via Variadic to match DSL signatures.
 * 
 * Scale Coefficient Representation:
 * - coe (coefficient) represents values in [0.5, 1.0) as fixed-point integers
 * - fixed_bits attribute specifies the number of bits: coe = value * 2^fixed_bits
 * - Default fixed_bits = 17, meaning coe ∈ [65536, 131071] for value ∈ [0.5, 1.0)
 * - Example: 0.625 * 2^17 = 81920 = 0b10100000000000000
 */

include "allo/Dialect/AlloDialect.td"
include "allo/Dialect/AlloTypes.td"
include "allo/Dialect/AlloAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Type constraints for quantization parameters based on pytorch.py conversion
// sign: i8 scalar or shaped (int8 sign, truly signed)
def SignType : AnyTypeOf<[I8, AnyTypeOf<[MemRefOf<[I8]>, RankedTensorOf<[I8]>]>]>;

// coe: i32 scalar or shaped (SEMANTICALLY uintN, stored as i32 due to MLIR limitation)
//      Range: usually 17-bit, [0x10000, 0x1FFFF] = [65536, 131071] represents normalized coeff [0.5, 1.0)
//      CRITICAL: In C++ lowering, MUST cast to uintN_t before arithmetic operations!
//                Directly using as signed i16 will cause incorrect comparisons/extensions.
//      Example:  uintN_t coeUnsigned = static_cast<uintN_t>(coeValue);
def CoeType : AnyTypeOf<[I32, AnyTypeOf<[MemRefOf<[I32]>, RankedTensorOf<[I32]>]>]>;

// rshift: i16 scalar or shaped (int16 shift amount, truly signed, can be negative for left shift)
def RshiftType : AnyTypeOf<[I16, AnyTypeOf<[MemRefOf<[I16]>, RankedTensorOf<[I16]>]>]>;

// zero: i8/i32 scalar or shaped (zero point, typically i8 for activations, i32 for accumulators)
def ZeroType : AnyTypeOf<[AnyInteger, AnyTypeOf<[MemRefOf<[AnyInteger]>, RankedTensorOf<[AnyInteger]>]>]>;

// stride: typically a tuple or small array, allow flexible integer types
def StrideType : AnyTypeOf<[AnyInteger, AnyTypeOf<[MemRefOf<[AnyInteger]>, RankedTensorOf<[AnyInteger]>]>]>;

//===----------------------------------------------------------------------===//
// Element Type Constraints for Quantized Operations
//===----------------------------------------------------------------------===//

// Integer element types for quantized tensors (activations, weights)
// Allows any signed integer type (i8, i16, i32, etc.) in memref or tensor
def IntElementType : AnyTypeOf<[
  MemRefOf<[AnySignlessInteger]>,
  RankedTensorOf<[AnySignlessInteger]>
]>;

// Float element types for floating-point tensors
// Allows any float type (f16, f32, f64, bf16) in memref or tensor
def FloatElementType : AnyTypeOf<[
  MemRefOf<[AnyFloat]>,
  RankedTensorOf<[AnyFloat]>
]>;

// Quantized MatMul: (output, lhs, rhs, x_scale_sign, x_scale_coe, x_scale_rshift, 
//                     y_scale_sign, y_scale_coe, y_scale_rshift,
//                     o_scale_sign, o_scale_coe, o_scale_rshift,
//                     o_scale_inv_sign, o_scale_inv_coe, o_scale_inv_rshift [, x_zero, y_zero, o_zero])
// Destination-passing style: output buffer is passed as first argument
def Allo_QMatMulOp : Allo_Op<"qmatmul", [AttrSizedOperandSegments]> {
  let summary = "Quantized matrix multiplication (destination-passing style)";
  let description = [{
    Performs quantized matmul: output = lhs @ rhs with per-tensor or per-channel quantization.
    Scale parameters are decomposed as: scale = sign * coe * 2^(-rshift).
    Optional inverse output scale (o_scale_inv) for reciprocal quantization: 1/o_scale.
    Optional zero points for asymmetric quantization: x_zero, y_zero, o_zero.
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands (output, lhs, rhs) must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$lhs,
    IntElementType:$rhs,
    SignType:$x_scale_sign,
    CoeType:$x_scale_coe,
    RshiftType:$x_scale_rshift,
    SignType:$y_scale_sign,
    CoeType:$y_scale_coe,
    RshiftType:$y_scale_rshift,
    SignType:$o_scale_sign,
    CoeType:$o_scale_coe,
    RshiftType:$o_scale_rshift,
    SignType:$o_scale_inv_sign,
    CoeType:$o_scale_inv_coe,
    RshiftType:$o_scale_inv_rshift,
    Optional<ZeroType>:$x_zero,
    Optional<ZeroType>:$y_zero,
    Optional<ZeroType>:$o_zero
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,` $x_scale_sign `,` $x_scale_coe `,` $x_scale_rshift `,`
        $y_scale_sign `,` $y_scale_coe `,` $y_scale_rshift `,`
        $o_scale_sign `,` $o_scale_coe `,` $o_scale_rshift `,`
        $o_scale_inv_sign `,` $o_scale_inv_coe `,` $o_scale_inv_rshift
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)` attr-dict `:` type($output) `,` type($lhs) `,` type($rhs) `,`
        type($x_scale_sign) `,` type($x_scale_coe) `,` type($x_scale_rshift) `,`
        type($y_scale_sign) `,` type($y_scale_coe) `,` type($y_scale_rshift) `,`
        type($o_scale_sign) `,` type($o_scale_coe) `,` type($o_scale_rshift) `,`
        type($o_scale_inv_sign) `,` type($o_scale_inv_coe) `,` type($o_scale_inv_rshift)
  }];
}

// Quantized MatMul with inverse sqrt(d) scaling
def Allo_QMatMulIsqrtDOp : Allo_Op<"qmatmul_isqrtd", [AttrSizedOperandSegments]> {
  let summary = "Quantized matmul with 1/sqrt(d) normalization (destination-passing style)";
  let description = [{
    Performs quantized matmul with inverse sqrt(d) scaling for attention mechanisms.
    Optional inverse output scale (o_scale_inv) for reciprocal quantization: 1/o_scale.
    Optional zero points for asymmetric quantization: x_zero, y_zero, o_zero.
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands (output, lhs, rhs) must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$lhs,
    IntElementType:$rhs,
    SignType:$x_scale_sign,
    CoeType:$x_scale_coe,
    RshiftType:$x_scale_rshift,
    SignType:$y_scale_sign,
    CoeType:$y_scale_coe,
    RshiftType:$y_scale_rshift,
    SignType:$o_scale_sign,
    CoeType:$o_scale_coe,
    RshiftType:$o_scale_rshift,
    SignType:$o_scale_inv_sign,
    CoeType:$o_scale_inv_coe,
    RshiftType:$o_scale_inv_rshift,
    Optional<ZeroType>:$x_zero,
    Optional<ZeroType>:$y_zero,
    Optional<ZeroType>:$o_zero
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,` $x_scale_sign `,` $x_scale_coe `,` $x_scale_rshift `,`
        $y_scale_sign `,` $y_scale_coe `,` $y_scale_rshift `,`
        $o_scale_sign `,` $o_scale_coe `,` $o_scale_rshift `,`
        $o_scale_inv_sign `,` $o_scale_inv_coe `,` $o_scale_inv_rshift
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)` attr-dict `:` type($output) `,` type($lhs) `,` type($rhs) `,`
        type($x_scale_sign) `,` type($x_scale_coe) `,` type($x_scale_rshift) `,`
        type($y_scale_sign) `,` type($y_scale_coe) `,` type($y_scale_rshift) `,`
        type($o_scale_sign) `,` type($o_scale_coe) `,` type($o_scale_rshift) `,`
        type($o_scale_inv_sign) `,` type($o_scale_inv_coe) `,` type($o_scale_inv_rshift)
  }];
}

// Quantized Conv2d: (output, inp, filter, stride, fscl_*, iscl_*, oscl_*, oscl_inv_*, wscl_*, bscl_* [, izr, ozr, bias])
def Allo_QConv2dOp : Allo_Op<"qconv2d", [AttrSizedOperandSegments]> {
  let summary = "Quantized 2D convolution (destination-passing style)";
  let description = [{
    Performs quantized 2D convolution with stride (compile-time constant attribute).
    Optional inverse output scale (oscl_inv) for reciprocal quantization: 1/oscl.
    Optional parameters: input_zero, output_zero, bias (integer bias tensor).
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$input,
    IntElementType:$filter,
    SignType:$fscl_sign, CoeType:$fscl_coe, RshiftType:$fscl_rshift,
    SignType:$iscl_sign, CoeType:$iscl_coe, RshiftType:$iscl_rshift,
    SignType:$oscl_sign, CoeType:$oscl_coe, RshiftType:$oscl_rshift,
    SignType:$oscl_inv_sign, CoeType:$oscl_inv_coe, RshiftType:$oscl_inv_rshift,
    SignType:$wscl_sign, CoeType:$wscl_coe, RshiftType:$wscl_rshift,
    Optional<SignType>:$bscl_sign, Optional<CoeType>:$bscl_coe, Optional<RshiftType>:$bscl_rshift,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    Optional<IntElementType>:$bias,
    DenseI64ArrayAttr:$stride
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `,` $filter `,`
        $fscl_sign `,` $fscl_coe `,` $fscl_rshift `,`
        $iscl_sign `,` $iscl_coe `,` $iscl_rshift `,`
        $oscl_sign `,` $oscl_coe `,` $oscl_rshift `,`
        $oscl_inv_sign `,` $oscl_inv_coe `,` $oscl_inv_rshift `,`
        $wscl_sign `,` $wscl_coe `,` $wscl_rshift
        (`,` $bscl_sign^ `,` $bscl_coe `,` $bscl_rshift)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))?
        (`bias` `=` $bias^ `:` type($bias))? `)`
        `stride` `=` $stride attr-dict `:` 
        type($output) `,` type($input) `,` type($filter) `,`
        type($fscl_sign) `,` type($fscl_coe) `,` type($fscl_rshift) `,`
        type($iscl_sign) `,` type($iscl_coe) `,` type($iscl_rshift) `,`
        type($oscl_sign) `,` type($oscl_coe) `,` type($oscl_rshift) `,`
        type($oscl_inv_sign) `,` type($oscl_inv_coe) `,` type($oscl_inv_rshift) `,`
        type($wscl_sign) `,` type($wscl_coe) `,` type($wscl_rshift)
        (`,` type($bscl_sign)^ `,` type($bscl_coe) `,` type($bscl_rshift))?
  }];
}

// Quantized Linear (Dense): (output, X, W, fscl_*, iscl_*, oscl_*, oscl_inv_*, wscl_*, bscl_* [, izr, ozr, bias])
def Allo_QLinearOp : Allo_Op<"qlinear", [AttrSizedOperandSegments]> {
  let summary = "Quantized linear (fully-connected) layer (destination-passing style)";
  let description = [{
    Performs quantized linear (dense) transformation: output = X @ W.T + bias.
    Optional inverse output scale (oscl_inv) for reciprocal quantization: 1/oscl.
    Optional parameters: input_zero, output_zero, bias (integer bias tensor).
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$input,
    IntElementType:$weight,
    SignType:$fscl_sign, CoeType:$fscl_coe, RshiftType:$fscl_rshift,
    SignType:$iscl_sign, CoeType:$iscl_coe, RshiftType:$iscl_rshift,
    SignType:$oscl_sign, CoeType:$oscl_coe, RshiftType:$oscl_rshift,
    SignType:$oscl_inv_sign, CoeType:$oscl_inv_coe, RshiftType:$oscl_inv_rshift,
    SignType:$wscl_sign, CoeType:$wscl_coe, RshiftType:$wscl_rshift,
    Optional<SignType>:$bscl_sign, Optional<CoeType>:$bscl_coe, Optional<RshiftType>:$bscl_rshift,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    Optional<IntElementType>:$bias
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `,` $weight `,` 
        $fscl_sign `,` $fscl_coe `,` $fscl_rshift `,`
        $iscl_sign `,` $iscl_coe `,` $iscl_rshift `,`
        $oscl_sign `,` $oscl_coe `,` $oscl_rshift `,`
        $oscl_inv_sign `,` $oscl_inv_coe `,` $oscl_inv_rshift `,`
        $wscl_sign `,` $wscl_coe `,` $wscl_rshift
        (`,` $bscl_sign^ `,` $bscl_coe `,` $bscl_rshift)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))?
        (`bias` `=` $bias^ `:` type($bias))? `)` attr-dict `:` 
        type($output) `,` type($input) `,` type($weight) `,`
        type($fscl_sign) `,` type($fscl_coe) `,` type($fscl_rshift) `,`
        type($iscl_sign) `,` type($iscl_coe) `,` type($iscl_rshift) `,`
        type($oscl_sign) `,` type($oscl_coe) `,` type($oscl_rshift) `,`
        type($oscl_inv_sign) `,` type($oscl_inv_coe) `,` type($oscl_inv_rshift) `,`
        type($wscl_sign) `,` type($wscl_coe) `,` type($wscl_rshift)
        (`,` type($bscl_sign)^ `,` type($bscl_coe) `,` type($bscl_rshift))?
  }];
}

// Quantized Add: (output, lhs, rhs, x_scale_*, y_scale_*, o_scale_*, o_scale_inv_* [, x_zero, y_zero, o_zero])
def Allo_QAddOp : Allo_Op<"qadd", [AttrSizedOperandSegments]> {
  let summary = "Quantized element-wise addition (destination-passing style)";
  let description = [{
    Performs quantized element-wise addition with requantization.
    Optional inverse output scale (o_scale_inv) for reciprocal quantization: 1/o_scale.
    Optional zero points for asymmetric quantization: x_zero, y_zero, o_zero.
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$lhs,
    IntElementType:$rhs,
    SignType:$x_scale_sign, CoeType:$x_scale_coe, RshiftType:$x_scale_rshift,
    SignType:$y_scale_sign, CoeType:$y_scale_coe, RshiftType:$y_scale_rshift,
    SignType:$o_scale_sign, CoeType:$o_scale_coe, RshiftType:$o_scale_rshift,
    SignType:$o_scale_inv_sign, CoeType:$o_scale_inv_coe, RshiftType:$o_scale_inv_rshift,
    Optional<ZeroType>:$x_zero,
    Optional<ZeroType>:$y_zero,
    Optional<ZeroType>:$o_zero
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,` 
        $x_scale_sign `,` $x_scale_coe `,` $x_scale_rshift `,`
        $y_scale_sign `,` $y_scale_coe `,` $y_scale_rshift `,`
        $o_scale_sign `,` $o_scale_coe `,` $o_scale_rshift `,`
        $o_scale_inv_sign `,` $o_scale_inv_coe `,` $o_scale_inv_rshift
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)` attr-dict `:` 
        type($output) `,` type($lhs) `,` type($rhs) `,`
        type($x_scale_sign) `,` type($x_scale_coe) `,` type($x_scale_rshift) `,`
        type($y_scale_sign) `,` type($y_scale_coe) `,` type($y_scale_rshift) `,`
        type($o_scale_sign) `,` type($o_scale_coe) `,` type($o_scale_rshift) `,`
        type($o_scale_inv_sign) `,` type($o_scale_inv_coe) `,` type($o_scale_inv_rshift)
  }];
}

// Integer GELU: (output, x, input_scale_*, gelu_scale_*, fused_scale_*, output_scale_*, output_scale_inv_* [, input_zero, output_zero])
def Allo_IntGELUOp : Allo_Op<"int_gelu", [AttrSizedOperandSegments]> {
  let summary = "Integer-arithmetic GELU activation (destination-passing style)";
  let description = [{
    Performs integer GELU activation using lookup tables and fixed-point arithmetic.
    Requires 4 scale groups:
    - input_scale: dequantize input to floating-point equivalent
    - gelu_scale: scale for GELU internal computation
    - fused_scale: precomputed gelu_scale / output_scale for efficiency
    - output_scale: quantize output back to integer
    - output_scale_inv: inverse of output_scale for reciprocal quantization (1/output_scale)
    Optional zero points for asymmetric quantization: input_zero, output_zero.
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$input,
    SignType:$input_scale_sign, CoeType:$input_scale_coe, RshiftType:$input_scale_rshift,
    SignType:$gelu_scale_sign, CoeType:$gelu_scale_coe, RshiftType:$gelu_scale_rshift,
    SignType:$fused_scale_sign, CoeType:$fused_scale_coe, RshiftType:$fused_scale_rshift,
    SignType:$output_scale_sign, CoeType:$output_scale_coe, RshiftType:$output_scale_rshift,
    SignType:$output_scale_inv_sign, CoeType:$output_scale_inv_coe, RshiftType:$output_scale_inv_rshift,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `,` 
        $input_scale_sign `,` $input_scale_coe `,` $input_scale_rshift `,`
        $gelu_scale_sign `,` $gelu_scale_coe `,` $gelu_scale_rshift `,`
        $fused_scale_sign `,` $fused_scale_coe `,` $fused_scale_rshift `,`
        $output_scale_sign `,` $output_scale_coe `,` $output_scale_rshift `,`
        $output_scale_inv_sign `,` $output_scale_inv_coe `,` $output_scale_inv_rshift
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)` attr-dict `:` 
        type($output) `,` type($input) `,`
        type($input_scale_sign) `,` type($input_scale_coe) `,` type($input_scale_rshift) `,`
        type($gelu_scale_sign) `,` type($gelu_scale_coe) `,` type($gelu_scale_rshift) `,`
        type($fused_scale_sign) `,` type($fused_scale_coe) `,` type($fused_scale_rshift) `,`
        type($output_scale_sign) `,` type($output_scale_coe) `,` type($output_scale_rshift) `,`
        type($output_scale_inv_sign) `,` type($output_scale_inv_coe) `,` type($output_scale_inv_rshift)
  }];
}

// Integer Softmax: (output, x, input_scale_*, softmax_scale_*, output_scale_*, output_scale_inv_*, fused_scale_* [, input_zero, output_zero])
def Allo_IntSoftmaxOp : Allo_Op<"int_softmax", [AttrSizedOperandSegments]> {
  let summary = "Integer-arithmetic Softmax (destination-passing style)";
  let description = [{
    Performs integer Softmax using exp LUT and fixed-point reciprocal approximation.
    Optional inverse output scale (output_scale_inv) for reciprocal quantization: 1/output_scale.
    Optional zero points for asymmetric quantization: input_zero, output_zero.
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$input,
    SignType:$input_scale_sign, CoeType:$input_scale_coe, RshiftType:$input_scale_rshift,
    SignType:$softmax_scale_sign, CoeType:$softmax_scale_coe, RshiftType:$softmax_scale_rshift,
    SignType:$output_scale_sign, CoeType:$output_scale_coe, RshiftType:$output_scale_rshift,
    SignType:$output_scale_inv_sign, CoeType:$output_scale_inv_coe, RshiftType:$output_scale_inv_rshift,
    Optional<SignType>:$fused_scale_sign, Optional<CoeType>:$fused_scale_coe, Optional<RshiftType>:$fused_scale_rshift,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `,` 
        $input_scale_sign `,` $input_scale_coe `,` $input_scale_rshift `,`
        $softmax_scale_sign `,` $softmax_scale_coe `,` $softmax_scale_rshift `,`
        $output_scale_sign `,` $output_scale_coe `,` $output_scale_rshift `,`
        $output_scale_inv_sign `,` $output_scale_inv_coe `,` $output_scale_inv_rshift
        (`,` $fused_scale_sign^ `,` $fused_scale_coe `,` $fused_scale_rshift)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)` attr-dict `:` 
        type($output) `,` type($input) `,`
        type($input_scale_sign) `,` type($input_scale_coe) `,` type($input_scale_rshift) `,`
        type($softmax_scale_sign) `,` type($softmax_scale_coe) `,` type($softmax_scale_rshift) `,`
        type($output_scale_sign) `,` type($output_scale_coe) `,` type($output_scale_rshift) `,`
        type($output_scale_inv_sign) `,` type($output_scale_inv_coe) `,` type($output_scale_inv_rshift)
        (`,` type($fused_scale_sign)^ `,` type($fused_scale_coe) `,` type($fused_scale_rshift))?
  }];
}

// Integer LayerNorm: (output, x, bias_int, input_scale_*, layernorm_scale_*, bias_scale_*, fused_scale_*, output_scale_*, output_scale_inv_* [, input_zero, output_zero])
def Allo_IntLayerNormOp : Allo_Op<"int_layernorm", [AttrSizedOperandSegments]> {
  let summary = "Integer-arithmetic Layer Normalization (destination-passing style)";
  let description = [{
    Performs integer LayerNorm with learned affine parameters (bias_int tensor).
    Requires 5 scale groups:
    - input_scale: dequantize input
    - layernorm_scale: scale for normalization
    - bias_scale: scale for bias addition
    - fused_scale: combined scale
    - output_scale: quantize output
    - output_scale_inv: inverse of output_scale for reciprocal quantization (1/output_scale)
    Optional zero points for asymmetric quantization: input_zero, output_zero.
    Output buffer must be pre-allocated by caller.
    
    Type constraints: All data operands must have integer element types.
  }];
  let arguments = (ins 
    IntElementType:$output,
    IntElementType:$input,
    IntElementType:$bias_int,
    SignType:$input_scale_sign, CoeType:$input_scale_coe, RshiftType:$input_scale_rshift,
    SignType:$layernorm_scale_sign, CoeType:$layernorm_scale_coe, RshiftType:$layernorm_scale_rshift,
    SignType:$bias_scale_sign, CoeType:$bias_scale_coe, RshiftType:$bias_scale_rshift,
    SignType:$fused_scale_sign, CoeType:$fused_scale_coe, RshiftType:$fused_scale_rshift,
    SignType:$output_scale_sign, CoeType:$output_scale_coe, RshiftType:$output_scale_rshift,
    SignType:$output_scale_inv_sign, CoeType:$output_scale_inv_coe, RshiftType:$output_scale_inv_rshift,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `,` $bias_int `,` 
        $input_scale_sign `,` $input_scale_coe `,` $input_scale_rshift `,`
        $layernorm_scale_sign `,` $layernorm_scale_coe `,` $layernorm_scale_rshift `,`
        $bias_scale_sign `,` $bias_scale_coe `,` $bias_scale_rshift `,`
        $fused_scale_sign `,` $fused_scale_coe `,` $fused_scale_rshift `,`
        $output_scale_sign `,` $output_scale_coe `,` $output_scale_rshift `,`
        $output_scale_inv_sign `,` $output_scale_inv_coe `,` $output_scale_inv_rshift
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)` attr-dict `:` 
        type($output) `,` type($input) `,` type($bias_int) `,`
        type($input_scale_sign) `,` type($input_scale_coe) `,` type($input_scale_rshift) `,`
        type($layernorm_scale_sign) `,` type($layernorm_scale_coe) `,` type($layernorm_scale_rshift) `,`
        type($bias_scale_sign) `,` type($bias_scale_coe) `,` type($bias_scale_rshift) `,`
        type($fused_scale_sign) `,` type($fused_scale_coe) `,` type($fused_scale_rshift) `,`
        type($output_scale_sign) `,` type($output_scale_coe) `,` type($output_scale_rshift) `,`
        type($output_scale_inv_sign) `,` type($output_scale_inv_coe) `,` type($output_scale_inv_rshift)
  }];
}

// ==================== Quantize / Dequantize Operations ====================
// These ops handle conversions between floating-point and integer representations
// at the boundaries of quantized and non-quantized computational regions.
//
// Quant Mode Encoding (2-bit integer):
//   Bit 0: 0 = symmetric, 1 = asymmetric
//   Bit 1: 0 = per-tensor, 1 = per-token
//   Values:
//     0 (0b00): symmetric, per-tensor
//     1 (0b01): asymmetric, per-tensor
//     2 (0b10): symmetric, per-token
//     3 (0b11): asymmetric, per-token

// Quantize: float -> int
// Performs: q = round(x / scale) [+ zero for asymmetric]
def Allo_QuantOp : Allo_Op<"quant", []> {
  let summary = "Quantize floating-point to integer (destination-passing style)";
  let description = [{
    Converts floating-point tensor to integer tensor using fixed-point scale.
    
    Quantization mode (quant_mode attribute, 2-bit encoding):
    - 0: symmetric, per-tensor
    - 1: asymmetric, per-tensor
    - 2: symmetric, per-token
    - 3: asymmetric, per-token
    
    Scale is decomposed as: scale = sign * coe * 2^(-rshift)
    
    Quantization formula:
    - Symmetric:  q = round(x / scale) = round(x * scale_inv)
    - Asymmetric: q = round(x / scale) + zero = round(x * scale_inv) + zero
    
    Optional zero point for asymmetric quantization.
    Output buffer must be pre-allocated by caller.
  }];
  let arguments = (ins 
    IntElementType:$output,      // Output: integer tensor (pre-allocated)
    FloatElementType:$input,     // Input: float tensor
    SignType:$scale_sign,        // Scale sign component
    CoeType:$scale_coe,          // Scale coefficient component
    RshiftType:$scale_rshift,    // Scale right-shift component
    Optional<ZeroType>:$zero,    // Zero point (for asymmetric mode)
    I8Attr:$quant_mode           // Quantization mode (0-3)
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `,` 
        $scale_sign `,` $scale_coe `,` $scale_rshift
        (`zero` `=` $zero^ `:` type($zero))? `)` 
        `quant_mode` `=` $quant_mode attr-dict `:` 
        type($output) `,` type($input) `,`
        type($scale_sign) `,` type($scale_coe) `,` type($scale_rshift)
  }];
}

// Dequantize: int -> float
// Performs: x = (q [- zero]) * scale
def Allo_DequantOp : Allo_Op<"dequant", []> {
  let summary = "Dequantize integer to floating-point (destination-passing style)";
  let description = [{
    Converts integer tensor to floating-point tensor using fixed-point scale.
    
    Quantization mode (quant_mode attribute, 2-bit encoding):
    - 0: symmetric, per-tensor
    - 1: asymmetric, per-tensor  
    - 2: symmetric, per-token
    - 3: asymmetric, per-token
    
    Scale is decomposed as: scale = sign * coe * 2^(-rshift)
    
    Dequantization formula:
    - Symmetric:  x = q * scale
    - Asymmetric: x = (q - zero) * scale
    
    Optional zero point for asymmetric quantization.
    Output buffer must be pre-allocated by caller.
  }];
  let arguments = (ins 
    FloatElementType:$output,    // Output: float tensor (pre-allocated)
    IntElementType:$input,       // Input: integer tensor
    SignType:$scale_sign,        // Scale sign component
    CoeType:$scale_coe,          // Scale coefficient component
    RshiftType:$scale_rshift,    // Scale right-shift component
    Optional<ZeroType>:$zero,    // Zero point (for asymmetric mode)
    I8Attr:$quant_mode           // Quantization mode (0-3)
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `,` 
        $scale_sign `,` $scale_coe `,` $scale_rshift
        (`zero` `=` $zero^ `:` type($zero))? `)` 
        `quant_mode` `=` $quant_mode attr-dict `:` 
        type($output) `,` type($input) `,`
        type($scale_sign) `,` type($scale_coe) `,` type($scale_rshift)
  }];
}
