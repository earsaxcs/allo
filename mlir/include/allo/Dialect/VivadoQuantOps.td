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
    ScaleType:$fused_scale,
    ScaleType:$o_scale,
    ScaleType:$o_scale_inv,
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
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed,

    // RHS (second input) layout tag.
    // Supported: "bhld", "blhd", "bhdl". Default is "bhld".
    DefaultValuedAttr<StrAttr, "\"bhld\"">:$rhs_layout,

    // RHS reduction (K) axis location (dimension index on RHS tensor/memref).
    // This indicates which RHS dimension is reduced over during matmul.
    // Default 2 matches historical defaults for rhs_layout="bhld".
    DefaultValuedAttr<I32Attr, "2">:$reduce_dim,

    // Layer type identification for optimization / backend layout decisions
    DefaultValuedAttr<StrAttr, "\"unknown\"">:$layer_type
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,`
      $x_scale `,` $y_scale `,` $fused_scale `,` $o_scale `,` $o_scale_inv
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)` attr-dict `:` 
        type($output) `,` type($lhs) `,` type($rhs) `,`
      type($x_scale) `,` type($y_scale) `,` type($fused_scale) `,` type($o_scale) `,` type($o_scale_inv)
  }];

  let hasVerifier = 1;
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
    
    **Layer Type Attribute:**
    The layer_type attribute identifies the role of this linear layer in the network:
    - "qkvgemm.proj_q": Query projection in multi-head attention
    - "qkvgemm.proj_k": Key projection in multi-head attention
    - "qkvgemm.proj_v": Value projection in multi-head attention
    - "attngemm.proj": Output projection after attention
    - "ffn.fc1": First FC layer in feed-forward network
    - "ffn.fc2": Second FC layer in feed-forward network
    - "classifier.dense": Classification head
    - "unknown": Default for unspecified layers
    
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
    ScaleType:$oscl_inv,
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
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed,
    
    // Layer type identification for optimization
    DefaultValuedAttr<StrAttr, "\"unknown\"">:$layer_type
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $weight `,`
        $fscl `,` $iscl `,` $oscl `,` $oscl_inv `,` $wscl
        (`,` $bscl^)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))?
        (`bias` `=` $bias^ `:` type($bias))? `)` attr-dict `:` 
        type($output) `,` type($input) `,` type($weight) `,`
        type($fscl) `,` type($iscl) `,` type($oscl) `,` type($oscl_inv) `,` type($wscl)
        (`,` type($bscl)^)?
  }];
  
    let hasVerifier = 1;
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
    ScaleType:$o_scale_inv,
    Optional<ZeroType>:$x_zero,
    Optional<ZeroType>:$y_zero,
    Optional<ZeroType>:$o_zero,
    
    DefaultValuedAttr<BoolAttr, "false">:$fuse_into_producer,
    DefaultValuedAttr<BoolAttr, "true">:$vectorize,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,`
        $x_scale `,` $y_scale `,` $o_scale `,` $o_scale_inv
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)`
        attr-dict `:` 
        type($output) `,` type($lhs) `,` type($rhs) `,`
        type($x_scale) `,` type($y_scale) `,` type($o_scale) `,` type($o_scale_inv)
  }];
  
  let hasVerifier = 1;
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
    ScaleType:$oscl_inv,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    
    DefaultValuedAttr<StrAttr, "\"lut\"">:$implementation,  // "lut" or "polynomial"
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,`
        $iscl `,` $gscl `,` $fscl `,` $oscl `,` $oscl_inv
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)`
        attr-dict `:` type($output) `,` type($input) `,`
        type($iscl) `,` type($gscl) `,` type($fscl) `,` type($oscl) `,` type($oscl_inv)
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
    ScaleType:$oscl_inv,
    Optional<ScaleType>:$fscl,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    I64Attr:$axis,
    
    DefaultValuedAttr<StrAttr, "\"lut\"">:$implementation,
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,`
        $iscl `,` $sscl `,` $oscl `,` $oscl_inv
        (`,` $fscl^)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)` attr-dict `:` 
        type($output) `,` type($input) `,`
        type($iscl) `,` type($sscl) `,` type($oscl) `,` type($oscl_inv)
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
    ScaleType:$oscl_inv,
    Optional<ZeroType>:$input_zero,
    Optional<ZeroType>:$output_zero,
    F32Attr:$eps,
    
    DefaultValuedAttr<StrAttr, "\"approx\"">:$rsqrt_method,  // "approx" or "lut"
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $bias_int `,`
        $iscl `,` $lscl `,` $bscl `,` $fscl `,` $oscl `,` $oscl_inv
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)`
        `eps` `=` $eps attr-dict `:` 
        type($output) `,` type($input) `,` type($bias_int) `,`
        type($iscl) `,` type($lscl) `,` type($bscl) `,` type($fscl) `,` type($oscl) `,` type($oscl_inv)
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
    ScaleType:$oscl_inv,
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
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $filter `,`
        $fscl `,` $iscl `,` $oscl `,` $oscl_inv `,` $wscl
        (`,` $bscl^)?
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))?
        (`bias` `=` $bias^ `:` type($bias))? `)`
        `stride` `=` $stride attr-dict `:` 
        type($output) `,` type($input) `,` type($filter) `,`
        type($fscl) `,` type($iscl) `,` type($oscl) `,` type($oscl_inv) `,` type($wscl)
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
    ScaleType:$fused_scale,
    ScaleType:$o_scale,
    ScaleType:$o_scale_inv,
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
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed,

    // RHS (second input) layout tag.
    // Supported: "bhld", "bhdl", "blhd", "bhdl". Default is "bhld".
    DefaultValuedAttr<StrAttr, "\"bhld\"">:$rhs_layout,

    // RHS reduction (K) axis location (dimension index on RHS tensor/memref).
    // This indicates which RHS dimension is reduced over during matmul.
    // Default 2 matches historical defaults for rhs_layout="bhld".
    DefaultValuedAttr<I32Attr, "2">:$reduce_dim,

    // Layer type identification for optimization / backend layout decisions
    DefaultValuedAttr<StrAttr, "\"unknown\"">:$layer_type
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $lhs `,` $rhs `,`
      $x_scale `,` $y_scale `,` $fused_scale `,` $o_scale `,` $o_scale_inv
        (`x_zero` `=` $x_zero^ `:` type($x_zero))?
        (`y_zero` `=` $y_zero^ `:` type($y_zero))?
        (`o_zero` `=` $o_zero^ `:` type($o_zero))? `)`
        attr-dict `:` 
        type($output) `,` type($lhs) `,` type($rhs) `,`
      type($x_scale) `,` type($y_scale) `,` type($fused_scale) `,` type($o_scale) `,` type($o_scale_inv)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Vivado Integer/Float Element Type Constraints
//===----------------------------------------------------------------------===//

// Integer element types for quantized tensors (activations, weights)
def Vivado_IntElementType : AnyTypeOf<[
  MemRefOf<[AnySignlessInteger]>,
  RankedTensorOf<[AnySignlessInteger]>
]>;

// Float element types for floating-point tensors
def Vivado_FloatElementType : AnyTypeOf<[
  MemRefOf<[AnyFloat]>,
  RankedTensorOf<[AnyFloat]>
]>;

//===----------------------------------------------------------------------===//
// Vivado Quantize Op
//===----------------------------------------------------------------------===//

def Vivado_QuantOp : Vivado_Op<"quant", []> {
  let summary = "Backend-specific quantize operation for Vivado/PYNQ";
  let description = [{
    Converts floating-point tensor to integer tensor using packed scale.
    
    This operation performs: q = round(x / scale) [+ zero for asymmetric]
    
    **Relationship to allo.quant:**
    - Inherits mathematical semantics from allo.quant
    - Uses packed 32-bit scale instead of separate sign/coe/rshift
    - Adds backend-specific attributes for hardware implementation
    
    **Quant Mode Encoding (2-bit):**
    - 0: symmetric, per-tensor
    - 1: asymmetric, per-tensor
    - 2: symmetric, per-token
    - 3: asymmetric, per-token
    
    **Lowering Target:**
    Will be lowered to PYNQ CPU-side (PS) operations.
  }];
  
  let arguments = (ins 
    Vivado_IntElementType:$output,    // Output: integer tensor (pre-allocated)
    Vivado_FloatElementType:$input,   // Input: float tensor
    ScaleType:$scale,                 // Packed scale (sign, coe, rshift in 32-bit)
    Optional<ZeroType>:$zero,         // Zero point (for asymmetric mode)
    
    // Backend-specific attributes
    I8Attr:$quant_mode,               // Quantization mode (0-3)
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $scale
        (`zero` `=` $zero^ `:` type($zero))? `)`
        `quant_mode` `=` $quant_mode attr-dict `:` 
        type($output) `,` type($input) `,` type($scale)
  }];
}

//===----------------------------------------------------------------------===//
// Vivado Dequantize Op
//===----------------------------------------------------------------------===//

def Vivado_DequantOp : Vivado_Op<"dequant", [/*AttrSizedOperandSegments*/]> {
  let summary = "Backend-specific dequantize operation for Vivado/PYNQ";
  let description = [{
    Converts integer tensor to floating-point tensor using packed scale.
    
    This operation performs: x = (q [- zero]) * scale
    
    **Relationship to allo.dequant:**
    - Inherits mathematical semantics from allo.dequant
    - Uses packed 32-bit scale instead of separate sign/coe/rshift
    - Adds backend-specific attributes for hardware implementation
    
    **Quant Mode Encoding (2-bit):**
    - 0: symmetric, per-tensor
    - 1: asymmetric, per-tensor
    - 2: symmetric, per-token
    - 3: asymmetric, per-token
    
    **Lowering Target:**
    Will be lowered to PYNQ CPU-side (PS) operations.
  }];
  
  let arguments = (ins 
    Vivado_FloatElementType:$output,  // Output: float tensor (pre-allocated)
    Vivado_IntElementType:$input,     // Input: integer tensor
    ScaleType:$scale,                 // Packed scale (sign, coe, rshift in 32-bit)
    Optional<ZeroType>:$zero,         // Zero point (for asymmetric mode)
    
    // Backend-specific attributes
    I8Attr:$quant_mode,               // Quantization mode (0-3)
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $scale
        (`zero` `=` $zero^ `:` type($zero))? `)`
        `quant_mode` `=` $quant_mode attr-dict `:` 
        type($output) `,` type($input) `,` type($scale)
  }];
}

//===----------------------------------------------------------------------===//
// Vivado ViT-specific Operations
//===----------------------------------------------------------------------===//

def Vivado_ViTGetFirstTokenOp : Vivado_Op<"vit_get_first_token", []> {
  let summary = "Extract first token from sequence for Vivado/PYNQ backend";
  let description = [{
    Backend-specific operation to extract the first token from a sequence.
    Input: [B, L, D], Output: [B, 1, D].
    
    **Backend-Specific Attributes:**
    - **is_transposed** (optional, default: false)
      Layout hint for hardware implementation.
      false: standard (B, L, D) layout
      true: transposed (B, D, L) layout
    
    **Lowering Target:**
    This op will be lowered to simple memory copy/slice operations.
  }];
  let arguments = (ins 
    AnyShaped:$output,
    AnyShaped:$input,
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $output `,` $input `)` attr-dict `:` type($output) `,` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Vivado Activation Layout Transpose Op
//===----------------------------------------------------------------------===//

def Vivado_ActivationLayoutTransposeOp : Vivado_Op<"activation_layout_transpose", []> {
  let summary = "Transpose activation tensor between (B,L,D) and (B,D,L) layouts";
  let description = [{
    Explicitly transposes activation tensors between standard layout and
    hardware-friendly transposed layout.
    
    **Purpose:**
    This op is inserted at model boundaries (entry/exit of transformer blocks)
    to convert between:
    - Standard layout: (Batch, SeqLen, EmbedDim) - used by embedding layers
    - Transposed layout: (Batch, EmbedDim, SeqLen) - used by hardware accelerators
    
    **Key Points:**
    1. Only used at boundaries - NOT inside transformer blocks
    2. Inside transformer blocks, all ops work in transposed layout (is_transposed=true)
    3. Batch dimension is preserved, only L and D dimensions are swapped
    
    **Usage Pattern:**
    ```
    [Embedding Layer] -> (B, L, D)
                            |
                vivado.activation_layout_transpose (to_transposed=true)
                            |
                         (B, D, L)
                            |
    [Transformer Blocks with is_transposed=true]
                            |
                         (B, D, L)
                            |
                vivado.activation_layout_transpose (to_transposed=false)
                            |
                         (B, L, D)
    [Output Layer]
    ```
    
    **Hardware Motivation:**
    Hardware matmul uses (K,M) x (K,N) = (M,N) format where reduce dimension K
    is exposed in row direction. Transposed layout aligns activations for this.
  }];
  
  let arguments = (ins 
    AnyShaped:$output,    // Output tensor with transposed layout
    AnyShaped:$input,     // Input tensor with original layout
    
    // Direction of transpose
    // true: (B, L, D) -> (B, D, L) [entering transformer blocks]
    // false: (B, D, L) -> (B, L, D) [exiting transformer blocks]
    BoolAttr:$to_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `)` `to_transposed` `=` $to_transposed attr-dict `:` 
    type($output) `,` type($input)
  }];
  
  // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Vivado Activation Dynamic Padding Op
//===----------------------------------------------------------------------===//

def Vivado_ActivationDynamicPadOp : Vivado_Op<"activation_dynamic_pad", []> {
  let summary = "Pad/unpad activation last dimension with zeros";
  let description = [{
    Inserts a boundary-only dynamic padding (zero-padding) for activation tensors.

    **Goal:**
    Align the *last* (contiguous) dimension of an activation buffer to a multiple
    of the hardware alignment unit (e.g. InstrConfig::kTileSize).

    **Usage Pattern:**
    This op is intended to be inserted ONLY at the beginning and end of a
    quantized region, so that intermediate buffers keep a padded shape without
    requiring per-op dynamic padding.

    - to_padded = true: copy input into output and fill padded tail with 0
    - to_padded = false: copy the non-padded prefix back to output

    **Notes:**
    - Rank can be 2/3/4; padding applies to the last dimension.
    - This is a semantic boundary op; lowering may map it to nested loops or
      memref copies depending on backend.
  }];

  let arguments = (ins
    AnyShaped:$output,
    AnyShaped:$input,

    // Direction:
    // true: unpadded -> padded (zero-fill tail)
    // false: padded -> unpadded (drop tail)
    BoolAttr:$to_padded
  );

  let results = (outs);

  let assemblyFormat = [{
    `(` $output `,` $input `)` `to_padded` `=` $to_padded attr-dict `:`
    type($output) `,` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Vivado Scale Conversion Op
//===----------------------------------------------------------------------===//

def Vivado_ScaleConversionOp : Vivado_Op<"scale_conversion", [AttrSizedOperandSegments]> {
  let summary = "Convert quantization scale between different modes";
  let description = [{
    Converts tensor quantization from one scale mode to another.
    
    **Purpose:**
    This operation bridges quantization scale mismatches between operations.
    For example, LayerNorm may output per-channel quantization, but the next
    operation may expect per-token quantization. This op performs the requantization
    to match scale modes.
    
    **Scale Modes:**
    - 0 (kPerTensor): Single scale for entire tensor
    - 1 (kPerChannel): Separate scale per channel (output dimension)
    - 2 (kPerToken): Separate scale per token (sequence dimension)
    
    **Semantics:**
    1. Dequantize input: x_float = (x_int - input_zero) * input_scale
    2. Requantize output: y_int = round(x_float / output_scale) + output_zero
    
    **Example:**
    LayerNorm output: per-channel scale (scale per D dimension)
    Next op expectation: per-token scale (scale per L dimension)
    This op performs: (B,L,D) with per-channel -> (B,L,D) with per-token
    
    **Usage Pattern:**
    ```mlir
    vivado.int_layernorm(...) -> %layernorm_out with per-channel scale
                                      |
    vivado.scale_conversion(%output, %layernorm_out, 
                            %input_scale, %output_scale)
      {input_scale_mode = 1 : i8,    // per-channel
       output_scale_mode = 2 : i8}   // per-token
                                      |
    vivado.qmatmul(...) expects per-token scale
    ```
  }];
  
  let arguments = (ins 
    AnyShaped:$output,              // Output: requantized tensor (pre-allocated)
    AnyShaped:$input,               // Input: quantized tensor to convert
    ScaleType:$input_scale,         // Input scale (packed format)
    ScaleType:$output_scale,        // Output scale (packed format)
    Optional<ZeroType>:$input_zero, // Input zero point (for asymmetric)
    Optional<ZeroType>:$output_zero,// Output zero point (for asymmetric)
    
    // Scale mode attributes
    I8Attr:$input_scale_mode,       // 0=per-tensor, 1=per-channel, 2=per-token
    I8Attr:$output_scale_mode,      // 0=per-tensor, 1=per-channel, 2=per-token
    
    // Backend-specific attributes
    DefaultValuedAttr<StrAttr, "\"8,24,8,16,16,0\"">:$scale_packing,
    DefaultValuedAttr<StrAttr, "\"Tail\"">:$scale_coe_mode,
    
    // Transpose mode attribute
    // - transpose_mode = false: op uses standard (non-transpose) semantics; is_transposed is ignored.
    // - transpose_mode = true: op interprets its inputs/logic in transpose-mode.
    // is_transposed indicates whether the *output memref* is in transposed layout.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_mode,
    DefaultValuedAttr<BoolAttr, "false">:$is_transposed
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `(` $output `,` $input `,` $input_scale `,` $output_scale
        (`input_zero` `=` $input_zero^ `:` type($input_zero))?
        (`output_zero` `=` $output_zero^ `:` type($output_zero))? `)`
        `input_scale_mode` `=` $input_scale_mode `,`
        `output_scale_mode` `=` $output_scale_mode attr-dict `:` 
        type($output) `,` type($input) `,` type($input_scale) `,` type($output_scale)
  }];
  
  // let hasVerifier = 1;
}