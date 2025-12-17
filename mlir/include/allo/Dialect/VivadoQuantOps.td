/*
 * Vivado Backend Quantized Ops
 * 
 * This dialect defines backend-specific quantized operations for Vivado HLS / PYNQ.
 * These ops inherit the mathematical semantics from allo.qmatmul but add backend-specific
 * attributes for hardware implementation (tiling, buffering, HLS pragmas, etc.).
 * 
 * Key Differences from Allo Quant Ops:
 * 1. Platform-Specific: Target Vivado HLS / PYNQ hardware constraints
 * 2. Implementation Hints: Contain attributes for tiling strategy, buffer allocation
 * 3. Optimization Hooks: Allow backend-specific transformations (loop pipelining, array partitioning)
 * 4. Intermediate IR: Bridge between platform-agnostic Allo ops and low-level PYNQ instructions
 * 
 * Lowering Pipeline:
 *   allo.qmatmul (frontend semantics)
 *       ↓ LowerAlloQuantToVivado
 *   vivado.qmatmul (backend semantics with attributes)
 *       ↓ LowerVivadoToPYNQInstr
 *   pynq.matmul_instr + pynq.data_transfer (hardware instructions)
 */

include "allo/Dialect/VivadoDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Type constraints for Vivado backend (packed scale format)
// ScaleType: 32-bit packed integer containing sign, coe, and rshift
// The packing layout is defined by ScalePackingAttr which is just a StrAttr
def ScaleType : AnyTypeOf<[I32, AnyTypeOf<[MemRefOf<[I32]>, RankedTensorOf<[I32]>]>]>;

// ScalePackingAttr: Defines the bit layout of the 32-bit packed scale
// Format: {sign_bits, sign_offset, coe_bits, coe_offset, rshift_bits, rshift_offset}
// Default layout: sign[31:24] (8 bits), rshift[23:16] (8 bits), coe[15:0] (16 bits)
// Example: "8,24,8,16,16,0" means:
//   - sign: 8 bit at position 24
//   - rshift: 8 bits at positions [23:16]
//   - coe: 16 bits at positions [15:0]
// def ScalePackingAttr : StrAttr;

// ScaleCoeMode: Defines the semantic type of the 'coe' field in the packed scale
// Options:
//   - "Full": Records the highest 1-bit of the fixed-point in [0.5, 1), e.g. records the whole part "1001110" of 0b0.1001110..., just ignores the 0b0.
//              For a 17-bit input (value * 2^17), takes the high 16 bits (value >> 1)
//   - "Tail": Records the variant part of the fixed-point in [0.5, 1), e.g. records the variant part "001110..." of 0b0.1001110..., ignores the 0b0.1 because it's always begins with 0b0.1
//              For a 17-bit input (value * 2^17), takes the low 16 bits (value & 0xFFFF)
// Default: "Tail" - matches the assumption that Allo ops provide coe as [0.5, 1) * 2^17
// define ScaleCoeMode : StrAttr;

def ZeroType : AnyTypeOf<[AnyInteger, AnyTypeOf<[MemRefOf<[AnyInteger]>, RankedTensorOf<[AnyInteger]>]>]>;

//===----------------------------------------------------------------------===//
// Vivado Quantized MatMul Op
//===----------------------------------------------------------------------===//

def Vivado_QMatMulOp : Vivado_Op<"qmatmul", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific quantized matrix multiplication for Vivado/PYNQ";
  let description = [{
    Performs quantized matrix multiplication with Vivado HLS backend optimizations.
    
    **Relationship to allo.qmatmul:**
    - Inherits ALL mathematical parameters (scales, zero points) unchanged
    - Adds backend-specific attributes for hardware implementation
    - Same destination-passing style and quantization semantics
    
    **Backend-Specific Attributes:**
    
    1. **tile_m, tile_n, tile_k** (optional, default: 32, 32, 32)
       - Tile sizes for blocking matrix multiplication
       - Constrained by PYNQ BRAM capacity (typically 32 rows × 32 cols)
       - Must be powers of 2 or multiples of 32 for efficient memory access
    
    2. **buffer_strategy** (optional, default: "auto")
       - "auto": Compiler automatically assigns buffer IDs
       - "manual": Use explicit buffer_ids attribute
       - "streaming": Enable streaming dataflow between operations
    
    3. **buffer_ids** (optional, array of 3 integers)
       - Explicit buffer assignments: [input_buf_id, weight_buf_id, output_buf_id]
       - Each ID must be in range [0, 7] (PYNQ has 8 on-chip buffers)
       - Used when buffer_strategy = "manual"
    
    4. **enable_bias** (optional, default: false)
       - Whether to fuse bias addition into matmul instruction
       - Maps to PYNQ hardware bias support (bias_tile_index in gen_matmul_instr)
    
    5. **transpose_weight** (optional, default: false)
       - Whether to transpose weight matrix during computation
       - Maps to transpose flag in pynq.matmul_instr
    
    6. **hls_pragmas** (optional, array of strings)
       - Custom HLS pragmas to apply (e.g., "PIPELINE II=1", "ARRAY_PARTITION")
       - Enables fine-grained performance tuning
    
    7. **accumulator_type** (optional, default: "i32")
       - Internal accumulator bit-width ("i32", "i64")
       - PYNQ hardware uses i32, but larger accumulator may be needed for large matrices
    
    8. **requant_mode** (optional, default: "inline")
       - "inline": Fuse requantization with matmul kernel
       - "separate": Generate separate requantization loop
       - "deferred": Defer requantization to next consumer
    
    **Example:**
    ```mlir
    vivado.qmatmul(%output, %lhs, %rhs,
                   %x_sign, %x_coe, %x_rshift,
                   %y_sign, %y_coe, %y_rshift,
                   %o_sign, %o_coe, %o_rshift)
      {
        tile_m = 32 : i32,
        tile_n = 32 : i32,
        tile_k = 32 : i32,
        buffer_strategy = "auto",
        transpose_weight = false,
        requant_mode = "inline",
        hls_pragmas = ["PIPELINE II=1", "ARRAY_PARTITION variable=output cyclic factor=4"]
      }
      : memref<128x128xi8>, memref<128x128xi8>, memref<128x128xi8>, ...
    ```
    
    **Lowering Target:**
    This op will be lowered to:
    1. Affine tiling loops (for %i = 0 to M step tile_m ...)
    2. pynq.data_transfer ops (host ↔ device DMA)
    3. pynq.matmul_instr (hardware instruction)
    4. Requantization loops (apply scale, clamp to i8)
  }];
  
  let arguments = (ins 
    // Same operands as allo.qmatmul (preserve semantics)
    AnyShaped:$output,
    AnyShaped:$lhs,
    AnyShaped:$rhs,
    ScaleType:$x_scale,
    ScaleType:$y_scale,
    ScaleType:$o_scale,
    Optional<ZeroType>:$x_zero,
    Optional<ZeroType>:$y_zero,
    Optional<ZeroType>:$o_zero,
    
    // Backend-specific attributes
    DefaultValuedAttr<I32Attr, "32">:$tile_m,
    DefaultValuedAttr<I32Attr, "32">:$tile_n,
    DefaultValuedAttr<I32Attr, "32">:$tile_k,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$buffer_strategy,
    OptionalAttr<DenseI64ArrayAttr>:$buffer_ids,
    DefaultValuedAttr<BoolAttr, "false">:$enable_bias,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_weight,
    OptionalAttr<StrArrayAttr>:$hls_pragmas,
    DefaultValuedAttr<StrAttr, "\"i32\"">:$accumulator_type,
    DefaultValuedAttr<StrAttr, "\"inline\"">:$requant_mode,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,`
        $x_scale `,` $y_scale `,` $o_scale
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)` attr-dict `:` 
        type($output) `,` type($lhs) `,` type($rhs) `,`
        type($x_scale) `,` type($y_scale) `,` type($o_scale)
  }];
  
  // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Vivado Quantized Linear Op
//===----------------------------------------------------------------------===//

def Vivado_QLinearOp : Vivado_Op<"qlinear", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific quantized linear layer for Vivado/PYNQ";
  let description = [{
    Quantized linear (fully-connected) layer: output = input @ weight.T + bias
    
    **Note:** Linear is semantically a MatMul with optional bias. This op
    provides a dedicated representation for linear layers to enable specific
    optimizations (e.g., weight pre-transposition, bias fusion).
    
    **Attributes:** Same as vivado.qmatmul, with additional:
    - **bias_buffer_id** (optional): Explicit buffer for bias (default: buffer 7)
    - **fuse_bias** (optional): Whether to fuse bias into matmul instruction
    
    **Lowering:** Can be lowered to vivado.qmatmul + bias handling, or directly
    to PYNQ instructions with bias support.
  }];
  
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$input,
    AnyShaped:$weight,
    ScaleType:$fscl,
    ScaleType:$iscl,
    ScaleType:$oscl,
    ScaleType:$wscl,
    Optional<ScaleType>:$bscl,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    Optional<AnyShaped>:$bias,
    
    // Backend-specific attributes (same as qmatmul)
    DefaultValuedAttr<I32Attr, "32">:$tile_m,
    DefaultValuedAttr<I32Attr, "32">:$tile_n,
    DefaultValuedAttr<I32Attr, "32">:$tile_k,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$buffer_strategy,
    OptionalAttr<DenseI64ArrayAttr>:$buffer_ids,
    DefaultValuedAttr<I32Attr, "7">:$bias_buffer_id,
    DefaultValuedAttr<BoolAttr, "true">:$fuse_bias,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_weight,
    OptionalAttr<StrArrayAttr>:$hls_pragmas,
    DefaultValuedAttr<StrAttr, "\"i32\"">:$accumulator_type,
    DefaultValuedAttr<StrAttr, "\"inline\"">:$requant_mode,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $weight `,`
        $fscl `,` $iscl `,` $oscl `,` $wscl
        (`,` $bscl^)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))?
        (`bias` `=` $bias^ `:` type($bias))? `)` attr-dict `:` 
        type($output) `,` type($input) `,` type($weight) `,`
        type($fscl) `,` type($iscl) `,` type($oscl) `,` type($wscl)
        (`,` type($bscl)^)?
  }];
  
  // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Vivado Quantized Add Op
//===----------------------------------------------------------------------===//

def Vivado_QAddOp : Vivado_Op<"qadd", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific quantized element-wise addition";
  let description = [{
    Quantized element-wise addition: output = lhs + rhs
    
    **Hardware Mapping:**
    - Simple element-wise operation, no tiling needed
    - Can be fused with preceding matmul (add bias to output buffer)
    - Typically lowered to CPU/scalar code, not PYNQ matmul accelerator
    
    **Attributes:**
    - **fuse_into_producer** (optional): Try to fuse into preceding matmul as bias
    - **vectorize** (optional): Enable SIMD vectorization
  }];
  
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$lhs,
    AnyShaped:$rhs,
    ScaleType:$x_scale,
    ScaleType:$y_scale,
    ScaleType:$o_scale,
    Optional<ZeroType>:$x_zero,
    Optional<ZeroType>:$y_zero,
    Optional<ZeroType>:$o_zero,
    
    DefaultValuedAttr<BoolAttr, "false">:$fuse_into_producer,
    DefaultValuedAttr<BoolAttr, "true">:$vectorize,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,`
        $x_scale `,` $y_scale `,` $o_scale
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)`
        attr-dict `:` 
        type($output) `,` type($lhs) `,` type($rhs) `,`
        type($x_scale) `,` type($y_scale) `,` type($o_scale)
  }];
}

//===----------------------------------------------------------------------===//
// Vivado Integer Activation Ops (GELU, Softmax, LayerNorm)
//===----------------------------------------------------------------------===//

def Vivado_IntGELUOp : Vivado_Op<"int_gelu", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific integer GELU activation";
  let description = [{
    Integer GELU approximation for quantized networks.
    
    **Hardware Mapping:**
    - Lookup table (LUT) based approximation
    - Or polynomial approximation with limited precision
    - Lowered to scalar loops or BRAM-based LUT
    
    **Scale Parameters (4 groups):**
    - iscl (input_scale): dequantize input
    - gscl (gelu_scale): GELU internal computation
    - fscl (fused_scale): precomputed gelu_scale / output_scale
    - oscl (output_scale): quantize output
  }];
  
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$input,
    ScaleType:$iscl,
    ScaleType:$gscl,
    ScaleType:$fscl,
    ScaleType:$oscl,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    
    DefaultValuedAttr<StrAttr, "\"lut\"">:$implementation,  // "lut" or "polynomial"
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,`
        $iscl `,` $gscl `,` $fscl `,` $oscl
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)`
        attr-dict `:` type($output) `,` type($input) `,`
        type($iscl) `,` type($gscl) `,` type($fscl) `,` type($oscl)
  }];
}

def Vivado_IntSoftmaxOp : Vivado_Op<"int_softmax", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific integer Softmax";
  let description = [{
    Integer Softmax approximation using fixed-point arithmetic.
    Typically lowered to: max subtraction + exp LUT + division/normalization.
  }];
  
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$input,
    ScaleType:$iscl,
    ScaleType:$sscl,
    ScaleType:$oscl,
    Optional<ScaleType>:$fscl,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    I64Attr:$axis,
    
    DefaultValuedAttr<StrAttr, "\"lut\"">:$implementation,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,`
        $iscl `,` $sscl `,` $oscl
        (`,` $fscl^)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)` attr-dict `:` 
        type($output) `,` type($input) `,`
        type($iscl) `,` type($sscl) `,` type($oscl)
        (`,` type($fscl)^)?
  }];
}

def Vivado_IntLayerNormOp : Vivado_Op<"int_layernorm", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific integer LayerNorm";
  let description = [{
    Integer LayerNorm: (x - mean) * gamma / sqrt(variance + eps) + beta
    Uses fixed-point arithmetic with reciprocal sqrt approximation.
    Requires 5 scale groups: input, layernorm, bias, fused, output.
  }];
  
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$input,
    AnyShaped:$bias_int,
    ScaleType:$iscl,
    ScaleType:$lscl,
    ScaleType:$bscl,
    ScaleType:$fscl,
    ScaleType:$oscl,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    F32Attr:$eps,
    
    DefaultValuedAttr<StrAttr, "\"approx\"">:$rsqrt_method,  // "approx" or "lut"
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $bias_int `,`
        $iscl `,` $lscl `,` $bscl `,` $fscl `,` $oscl
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)`
        `eps` `=` $eps attr-dict `:` 
        type($output) `,` type($input) `,` type($bias_int) `,`
        type($iscl) `,` type($lscl) `,` type($bscl) `,` type($fscl) `,` type($oscl)
  }];
}

//===----------------------------------------------------------------------===//
// Vivado Quantized Conv2d Op
//===----------------------------------------------------------------------===//

def Vivado_QConv2dOp : Vivado_Op<"qconv2d", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific quantized 2D convolution";
  let description = [{
    Performs quantized 2D convolution with stride (compile-time constant attribute).
    Optional parameters: input_zero, output_zero, bias (integer bias tensor).
    Output buffer must be pre-allocated by caller.
    
    **Backend-Specific Attributes:**
    - **tile_h, tile_w, tile_c** (optional): Tiling factors
    - **buffer_strategy**, **hls_pragmas**, etc. (same as qmatmul)
  }];
  
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$input,
    AnyShaped:$filter,
    ScaleType:$fscl,
    ScaleType:$iscl,
    ScaleType:$oscl,
    ScaleType:$wscl,
    Optional<ScaleType>:$bscl,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    Optional<AnyShaped>:$bias,
    DenseI64ArrayAttr:$stride,
    
    // Backend-specific attributes
    DefaultValuedAttr<I32Attr, "32">:$tile_h,
    DefaultValuedAttr<I32Attr, "32">:$tile_w,
    DefaultValuedAttr<I32Attr, "32">:$tile_c,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$buffer_strategy,
    OptionalAttr<DenseI64ArrayAttr>:$buffer_ids,
    DefaultValuedAttr<BoolAttr, "false">:$enable_bias,
    OptionalAttr<StrArrayAttr>:$hls_pragmas,
    DefaultValuedAttr<StrAttr, "\"i32\"">:$accumulator_type,
    DefaultValuedAttr<StrAttr, "\"inline\"">:$requant_mode,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $filter `,`
        $fscl `,` $iscl `,` $oscl `,` $wscl
        (`,` $bscl^)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))?
        (`bias` `=` $bias^ `:` type($bias))? `)`
        `stride` `=` $stride attr-dict `:` 
        type($output) `,` type($input) `,` type($filter) `,`
        type($fscl) `,` type($iscl) `,` type($oscl) `,` type($wscl)
        (`,` type($bscl)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Vivado Quantized MatMul IsqrtD Op
//===----------------------------------------------------------------------===//

def Vivado_QMatMulIsqrtDOp : Vivado_Op<"qmatmul_isqrtd", [AttrSizedOperandSegments]> {
  let summary = "Backend-specific quantized matmul with 1/sqrt(d) scaling";
  let description = [{
    Performs quantized matmul with inverse sqrt(d) scaling.
    Used in attention mechanisms.
    
    **Attributes:** Same as qmatmul.
  }];
  
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$lhs,
    AnyShaped:$rhs,
    ScaleType:$x_scale,
    ScaleType:$y_scale,
    ScaleType:$o_scale,
    Optional<ZeroType>:$x_zero,
    Optional<ZeroType>:$y_zero,
    Optional<ZeroType>:$o_zero,
    
    // Backend-specific attributes
    DefaultValuedAttr<I32Attr, "32">:$tile_m,
    DefaultValuedAttr<I32Attr, "32">:$tile_n,
    DefaultValuedAttr<I32Attr, "32">:$tile_k,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$buffer_strategy,
    OptionalAttr<DenseI64ArrayAttr>:$buffer_ids,
    DefaultValuedAttr<BoolAttr, "false">:$enable_bias,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_weight,
    OptionalAttr<StrArrayAttr>:$hls_pragmas,
    DefaultValuedAttr<StrAttr, "\"i32\"">:$accumulator_type,
    DefaultValuedAttr<StrAttr, "\"inline\"">:$requant_mode,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,`
        $x_scale `,` $y_scale `,` $o_scale
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)`
        attr-dict `:` 
        type($output) `,` type($lhs) `,` type($rhs) `,`
        type($x_scale) `,` type($y_scale) `,` type($o_scale)
  }];
}

