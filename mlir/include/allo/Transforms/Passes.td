/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MLIR_PASSES
#define ALLO_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def LoopTransformation : Pass<"loop-opt", "ModuleOp"> {
  let summary = "Loop transformation pass";
  let constructor = "mlir::allo::createLoopTransformationPass()";
}

def DataPlacement : Pass<"data-placement", "ModuleOp"> {
  let summary = "Data placement pass";
  let constructor = "mlir::allo::createDataPlacementPass()";
}

def AnyWidthInteger : Pass<"anywidth-integer", "ModuleOp"> {
  let summary = "Transform anywidth-integer input to 64-bit";
  let constructor = "mlir::allo::createAnyWidthIntegerPass()";
}

def MoveReturnToInput : Pass<"return-to-input", "ModuleOp"> {
  let summary = "Move return values to input argument list";
  let constructor = "mlir::allo::createMoveReturnToInputPass()";
}

def LegalizeCast : Pass<"legalize-cast", "ModuleOp"> {
  let summary = "Legalize cast operations";
  let constructor = "mlir::allo::createLegalizeCastPass()";
}

def RemoveStrideMap : Pass<"remove-stride-map", "ModuleOp"> {
  let summary = "Remove stride map from partitioned memref";
  let constructor = "mlir::allo::createRemoveStrideMapPass()";
}

def MemRefDCE : Pass<"memref-dce", "ModuleOp"> {
  let summary = "Remove MemRefs that are never loaded from";
  let constructor = "mlir::allo::createMemRefDCEPass()";
}

def MergeSubviewAndCopy : Pass<"merge-subview-and-copy", "ModuleOp"> {
  let summary = "Merge subview and copy operations for dual Concat";
  let constructor = "mlir::allo::createMergeSubviewAndCopyPass()";
}

def HoistGetGlobal : Pass<"hoist-get-global", "ModuleOp"> {
  let summary = "Hoist memref.get_global ops to the top of their containing blocks";
  let description = [{
    This pass moves all `memref.get_global` operations to the beginning of the
    block that contains them, preserving their relative order.

    This is similar in spirit to `pynq-hoist-buffer-alloc`, but operates at
    block granularity and targets global constant materialization.
  }];
  let constructor = "mlir::allo::createHoistGetGlobalPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PYNQ Backend Passes
//===----------------------------------------------------------------------===//

def PYNQHoistBufferAlloc : Pass<"pynq-hoist-buffer-alloc", "ModuleOp"> {
  let summary = "Hoist allocation ops to function entry and dedupe same-ID buffers";
  let description = [{
    This pass moves allocation-like ops to the entry block of each function.
    It primarily hoists `pynq.buffer_alloc` and also hoists allocator-inserted
    spill slots (`memref.alloc` tagged as spill slots).

    When run after `pynq-buffer-allocation`, it additionally deduplicates
    multiple `pynq.buffer_alloc` ops that allocate the same physical buffer ID,
    rewriting all uses to a single SSA value.
    
    Compilation Flow:
    ```
    // Before: allocs scattered near first use
    func.func @kernel(...) {
      %buf0 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      pynq.data_transfer %A -> %buf0
      %buf1 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      pynq.data_transfer %B -> %buf1
      ...
    }
    
    // After: all allocs at function entry
    func.func @kernel(...) {
      %buf0 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      %buf1 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      pynq.data_transfer %A -> %buf0
      pynq.data_transfer %B -> %buf1
      ...
    }
    ```
    
    This pass is typically run after PYNQBufferAllocation.
  }];
  let constructor = "mlir::allo::createPYNQHoistBufferAllocPass()";
  
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def PYNQBufferAllocation : Pass<"pynq-buffer-allocation", "ModuleOp"> {
  let summary = "Assign physical buffer IDs to virtual buffers";
  let description = [{
    This pass assigns physical buffer IDs to virtual pynq.buffer_alloc ops
    using linear scan register allocation. Virtual buffers (with id=?) are
    transformed into physical buffers (with id=0,1,2,...).
    
    The pass performs:
    1. Liveness analysis of buffer usage
    2. Linear scan allocation for buffer IDs (0 to numBuffers-1)
    3. Buffer merging when lifetimes don't overlap
    4. Spill insertion when buffers are exhausted (if enabled)
    
    Compilation Flow:
    ```
    // Before: virtual buffers (id=?)
    func.func @kernel(...) {
      %buf0 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      %buf1 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      %buf2 = pynq.buffer_alloc : !pynq.buffer<i32, ?, 16384>
      ...
    }
    
    // After: physical buffers (id=0,1,2,...)
    func.func @kernel(...) {
      %buf0 = pynq.buffer_alloc : !pynq.buffer<i8, 0, 4096>
      %buf1 = pynq.buffer_alloc : !pynq.buffer<i8, 1, 4096>
      %buf2 = pynq.buffer_alloc : !pynq.buffer<i32, 2, 16384>
      ...
    }
    ```
    
    If two virtual buffers have non-overlapping lifetimes, they may be
    merged to use the same physical buffer ID.
    
    Hardware constraints are configurable via pass options:
    - numBuffers: Number of on-chip buffers (default: 8)
    - bufferCapacity: Default buffer capacity in bytes (default: 1024)
    - enableSpilling: Whether to enable buffer spilling (default: false)
  }];
  let constructor = "mlir::allo::createPYNQBufferAllocationPass()";
  
  let options = [
    Option<"numBuffers", "num-buffers", "unsigned", "8",
           "Number of on-chip buffers available">,
    Option<"bufferCapacity", "buffer-capacity", "int64_t", "1024",
           "Default buffer capacity in bytes">,
    Option<"enableSpilling", "enable-spilling", "bool", "false",
           "Enable buffer spilling when all buffers are occupied">
  ];
  
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect"
  ];
}

def PYNQSimplifyHostTransfers : Pass<"pynq-simplify-host-transfers", "ModuleOp"> {
  let summary = "(WIP) Simplify redundant host memref transfers between PYNQ buffers";
  let description = [{
    This pass is planned to reduce unnecessary buffer->memref->buffer round-trips
    after `lower-vivado-to-pynq`, keeping data in `!pynq.buffer` whenever possible.

    Current status: scaffolding / dry-run (no IR rewrite by default).
  }];
  let constructor = "mlir::allo::createPYNQSimplifyHostTransfersPass()";
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def PYNQElideRedundantCopies : Pass<"pynq-elide-redundant-copies", "ModuleOp"> {
  let summary = "Elide redundant pynq.copy ops by forwarding buffer reads";
  let description = [{
    This pass performs copy forwarding/copy propagation for `!pynq.buffer`.
    It is intended to run after host-transfer simplifications, and can reduce
    intermediate buffer usage by rewriting users of a buffer to directly use
    the buffer that most recently defined its contents.

    Current status: conservative, block-local forwarding (CFG dataflow TBD).
  }];
  let constructor = "mlir::allo::createPYNQElideRedundantCopiesPass()";
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect"
  ];
}

def PYNQScheduleOps : Pass<"pynq-schedule-ops", "ModuleOp"> {
  let summary = "Conservatively schedule pynq ops to reduce peak live buffers";
  let description = [{
    This pass performs conservative block-local scheduling of a subset of
    PYNQ operations to reduce the estimated peak number of simultaneously-live
    `!pynq.buffer` values.

    Correctness-first design:
    - Only reorders within a single basic block.
    - Only schedules within contiguous windows of known PYNQ ops.
    - Treats unknown ops, `pynq.sync`, and terminators as barriers.
    - Preserves SSA def-use dependencies and per-buffer write ordering.

    Default mode is dry-run (analysis only).
  }];
  let constructor = "mlir::allo::createPYNQScheduleOpsPass()";
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect"
  ];
}

def PYNQAdjustScaleRShift : Pass<"pynq-adjust-scale-rshift", "ModuleOp"> {
  let summary = "Adjust packed-scale rshift for PYNQ ops using per-op desired offsets";
  let description = [{
    This pass visits PYNQ compute ops that carry bound packed-scale globals
    (excluding `pynq.quant`/`pynq.dequant`) and adjusts packed `rshift`
    according to each scale's desired offset.

    Adjustment rule:
    - delta = desired_offset - innate_rshift_bias

    The pass unpacks each i32 packed-scale element using the canonical target
    packing/coefficient mode, applies `delta` to `rshift`, then packs values
    back to the same target format.

    It is intended to run after `repack-vivado-scales` and before
    `pynq-mid-lower`.
  }];
  let constructor = "mlir::allo::createPYNQAdjustScaleRShiftPass()";
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def PYNQMidLower : Pass<"pynq-mid-lower", "ModuleOp"> {
  let summary = "Mid-lower PYNQ ops: pack scale/bias to FIFO transfers and lower to instr ops";
  let description = [{
    This pass separates scale/bias transfers from compute operations.

    It:
    - Walks PYNQ compute ops in IR order and collects their bound scale (+bias for layernorm)
      from memref.global constants.
    - Preprocesses scales (chunking + half-interleave, plus block-wise stacking for multi-scale vector ops).
    - Packs scales into per-transfer memref.global blobs and inserts pynq.data_transfer_instr to stream
      them to the hardware FIFOs (direction=2 for matrix FIFO, direction=3 for vector FIFO).
    - Lowers pynq.matmul and vector ops to their corresponding *_instr ops, removing scale bindings.
    - Lowers pynq.copy to pynq.data_transfer_instr for host<->device transfers.

    Debug options:
    - debug-scale-pack: verbose printing for scale/bias packing (raw values, pairing order, packed bytes, segments/transfers).
    - debug-op-convert: printing for op conversion to *_instr (buffer ids, tile_count/reduceK, and key instruction fields).

    This pass must run after pynq-buffer-allocation; any virtual !pynq.buffer triggers an error.
  }];
  let constructor = "mlir::allo::createPYNQMidLowerPass()";
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PYNQReturnMemrefToOutParam : Pass<"pynq-return-memref-to-out-param", "ModuleOp"> {
  let summary = "Rewrite single-memref function returns into explicit output arguments";
  let description = [{
    For functions that satisfy strict preconditions, this pass rewrites
    `func.func` signatures from returning a single memref to returning `void`
    with one additional output memref argument.

    It also rewrites corresponding `func.call` sites by creating an extra
    `memref.alloc` as output buffer, passing it as the new trailing argument,
    and replacing old call result uses with that allocated buffer.

    Functions/calls that do not satisfy preconditions are reported with
    concrete reasons.
  }];
  let constructor = "mlir::allo::createPYNQReturnMemrefToOutParamPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def PYNQOptimizeSync : Pass<"pynq-optimize-sync", "ModuleOp"> {
  let summary = "Elide redundant pynq.sync ops using conservative dependency checks";
  let description = [{
    This pass removes redundant `pynq.sync` operations to reduce unnecessary
    stalls.

    Optimization rule (conservative):
    - For a `pynq.sync` between two async PYNQ hardware-issuing windows, the sync
      can be removed if the next operation after the sync has no potential
      buffer hazards (RAW/WAR/WAW) with operations issued since the previous
      kept sync.
    - If buffer IDs or directions are not statically known, the pass keeps the
      sync.

    This pass is intended to run after `pynq-mid-lower`.
  }];
  let constructor = "mlir::allo::createPYNQOptimizeSyncPass()";
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def PYNQOptimizeSubviewGlobals : Pass<"pynq-optimize-subview-globals", "ModuleOp"> {
  let summary = "Optimize subview on static weight/bias globals";
  let description = [{
    This pass simplifies memref.subview on globals that represent static
    weight/bias data at PYNQ level.

    - If a subview is an identity (offsets all zero, sizes cover the full
      source, strides all one), the subview is removed.
    - Otherwise, a new memref.global is materialized for the slice and the
      subview is replaced by a direct memref.get_global to the new global.

    This avoids non-contiguous subviews for static data in later stages.
  }];
  let constructor = "mlir::allo::createPYNQOptimizeSubviewGlobalsPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Vivado Backend Passes
//===----------------------------------------------------------------------===//

def VivadoQLinearKSplit : Pass<"vivado-qlinear-k-split", "ModuleOp"> {
  let summary = "Split vivado.qlinear ops with K dimension exceeding tile size";
  let description = [{
    This pass handles the case where the K dimension (reduction dimension) of
    vivado.qlinear operations exceeds the hardware tile size limit (TileConfig::kDefaultTileK).
    
    **Motivation:**
    In FC2 layers of transformer models, the weight matrix dimensions can be quite large.
    When K (the reduction/input dimension) exceeds the tile size limit:
    - For M and N dimensions: can be handled later during Vivado->PYNQ lowering as
      independent tiles without data dependencies
    - For K dimension: must be split early because partial results have data dependencies
      (each K-tile produces a partial sum that must be accumulated via QAdd)
    
    **Transformation:**
    ```mlir
    // Before: QLinear with K > TileK
    vivado.qlinear(%output, %input, %weight, ...) 
      : [B,L,K_large] x [OC,K_large] -> [B,L,OC]
    
    // After: Multiple QLinear + QAdd chain
    %input_0 = memref.subview %input[..., 0:TileK]
    %weight_0 = memref.subview %weight[..., 0:TileK]
    %partial_0 = memref.alloc()
    vivado.qlinear(%partial_0, %input_0, %weight_0, ...)
    
    %input_1 = memref.subview %input[..., TileK:2*TileK]
    %weight_1 = memref.subview %weight[..., TileK:2*TileK]
    %partial_1 = memref.alloc()
    vivado.qlinear(%partial_1, %input_1, %weight_1, ...)
    
    // Accumulate via QAdd chain
    vivado.qadd(%output, %partial_0, %partial_1, %o_scale, %o_scale, ...)
    ```
    
    **Key Design Decisions:**
    1. QAdd uses original QLinear's o_scale (packed, not inv) for both inputs
       since all partial results share the same quantization scale.
    2. K dimension split is done here; M/N splits are deferred to PYNQ lowering.
    3. The is_transposed attribute determines K dimension location.
    
    This pass should run before LowerVivadoToPYNQ.
  }];
  let constructor = "mlir::allo::createVivadoQLinearKSplitPass()";
  
  let dependentDialects = [
    "mlir::allo::vivado::VivadoDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def RepackVivadoScales : Pass<"repack-vivado-scales", "ModuleOp"> {
  let summary = "Repack Vivado quantized op scale parameters";
  let description = [{
    This pass transforms packed scale parameters from one format to another
    according to global packing configuration strings. The primary use case is
    converting scale formats after lowering to Vivado ops.
    
    Transformation details:
    - Source format example: "8,24,8,16,16,0" (Tail mode)
      Layout: sign[31:24] | rshift[23:16] | coe[15:0]
    - Target format example: "0,22,6,16,16,0" (Full mode, sign-fused)
      Layout: rshift[27:22] | coe[21:0] (two's complement)
    
    The pass performs:
    1. Tail->Full mode conversion (adds explicit leading 1 to coefficient)
    2. Sign fusion (converts signed coefficient to two's complement)
    3. Bit width adjustments (e.g., 16->22 bits for coefficient)
    4. rshift correction for bit width changes
    
    Supports all Vivado quantized operations:
    - vivado.qmatmul, vivado.qlinear, vivado.qadd
    - vivado.int_gelu, vivado.int_softmax, vivado.int_layernorm
    - vivado.qconv2d, vivado.qmatmul_isqrtd
  }];
  let constructor = "mlir::allo::createRepackVivadoScalesPass()";
  let dependentDialects = ["mlir::allo::vivado::VivadoDialect"];
}

def InsertScaleConversionAfterLayerNorm : Pass<"insert-scale-conversion-after-layernorm", "ModuleOp"> {
  let summary = "Insert missing scale conversions after LayerNorm";
  let description = [{
    This pass inspects vivado.int_layernorm outputs and inserts a
    vivado.scale_conversion before every consumer whose quantization
    parameters expect a different scale mode (per-token vs per-channel).
    The goal is to automatically bridge the LayerNorm legacy per-channel
    output scales with downstream ops that may expect per-token scales.
  }];
  let constructor = "mlir::allo::createInsertScaleConversionAfterLayerNormPass()";
  let dependentDialects = [
    "mlir::allo::vivado::VivadoDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ToggleVivadoTranspose : Pass<"toggle-vivado-transpose", "ModuleOp"> {
  let summary = "Toggle is_transposed attribute for Vivado quantized operations";
  let description = [{
    This pass toggles the is_transposed attribute for all Vivado quantized operations
    (qlinear, qmatmul, qadd, etc.) in the module. This is useful for switching between
    different layout conventions (e.g., (B,L,D) vs (B,D,L) for activations).
    
    The pass modifies:
    - vivado.qlinear: toggles is_transposed attribute
    - vivado.qmatmul: toggles is_transposed attribute
    - vivado.qadd: toggles is_transposed attribute
    - vivado.int_gelu: toggles is_transposed attribute
    - vivado.int_softmax: toggles is_transposed attribute
    - vivado.int_layernorm: toggles is_transposed attribute
    - vivado.qconv2d: toggles is_transposed attribute (if applicable)
    
    When is_transposed changes from false to true:
    - Input layout: (B, L, D) → (B, D, L)
    - Weight layout: (OC, IC) → (IC, OC) for qlinear
    - Output layout: (B, L, OC) → (B, OC, L)
    
    This pass is typically used for:
    1. Adapting to different hardware layout requirements
    2. Testing performance of different memory layouts
    3. Interfacing with external frameworks that use different conventions
  }];
  let constructor = "mlir::allo::createToggleVivadoTransposePass()";
  let dependentDialects = ["mlir::allo::vivado::VivadoDialect"];
}

def VivadoMergeRedundantQuantDequant : Pass<"vivado-merge-redundant-quant-dequant", "ModuleOp"> {
  let summary = "Merge identical vivado.quant/vivado.dequant ops that share the same source and quant parameters";
  let description = [{
    This pass deduplicates boundary quantization ops inserted at multiple use-sites.

    It finds pairs of vivado.quant (or vivado.dequant) that:
    - Quantize/dequantize the same input value
    - Use the exact same scale/zero operands
    - Have the exact same attribute dictionary (including quant_mode, packing/coefficient mode, transpose flags, etc.)
    - Produce the same output buffer element type/shape

    If all parameters match, the pass rewrites later consumers to reuse the earlier
    quantized/dequantized buffer and removes the redundant quant/dequant op.

    IMPORTANT:
    - The pass is conservative: if it cannot prove that the buffer is only read after
      quant/dequant (e.g., unknown memory effects), it will NOT merge.
    - It must run before toggle-vivado-transpose so that the transpose toggling logic
      sees a minimized set of boundary ops.
  }];
  let constructor = "mlir::allo::createVivadoMergeRedundantQuantDequantPass()";
  let dependentDialects = [
    "mlir::allo::vivado::VivadoDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def VivadoPadding : Pass<"vivado-padding", "ModuleOp"> {
  let summary = "Pad weight/bias/activations to hardware tile-size alignment";
  let description = [{
    This pass performs the Vivado/PYNQ *padding* step only.

    **Alignment rule:**
    Pad the *last dimension* so that:
    padded_last_dim * element_bytes is a multiple of InstrConfig::kTileSize (bytes).
    Example: ?x197xi8 with kTileSize=32B -> ?x224xi8.

    **What gets padded:**
    - Weight globals (2D): pad last dim only, zero-filled
    - Bias globals (2D only): pad last dim only, zero-filled
    - Activations (rank 2/3/4): only adjust *types* (last dim) for buffers inside
      the quantized region; boundary copies are represented explicitly by
      vivado.activation_dynamic_pad inserted right after vivado.quant and right
      before vivado.dequant.

    **Important:**
    - This pass does NOT perform any transpose. Transpose is handled elsewhere.
    - IntLayerNorm bias_int must NOT be padded.
  }];
  let constructor = "mlir::allo::createVivadoPaddingPass()";
  let dependentDialects = [
    "mlir::allo::vivado::VivadoDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def VivadoSeparateBias : Pass<"vivado-separate-bias", "ModuleOp"> {
  let summary = "Separate fused bias from vivado.qlinear into explicit vivado.qadd";
  let description = [{
    This pass separates the fused bias from vivado.qlinear operations into
    separate vivado.qadd operations.
    
    **Motivation:**
    - Some hardware implementations don't support fused bias in matmul
    - Explicit bias operations allow different quantization parameters
    - Useful for debugging or analysis requiring explicit bias operations
    
    **Transformation:**
    ```mlir
    // Before: QLinear with fused bias
    vivado.qlinear(%output, %input, %weight, ..., %bscl, ..., %bias)
      {fuse_bias = true}
    
    // After: QLinear without bias + QAdd for bias
    %temp = memref.alloc : memref<...>
    vivado.qlinear(%temp, %input, %weight, ..., /*no bscl*/, ..., /*no bias*/)
      {fuse_bias = false}
    vivado.qadd(%output, %temp, %bias, %oscl, %bscl, %oscl, %oscl_inv, ...)
    ```
    
    **Scale Mapping:**
    - QAdd's x_scale = QLinear's output scale (oscl)
    - QAdd's y_scale = QLinear's bias scale (bscl)
    - QAdd's o_scale = QLinear's output scale (oscl)
    
    **Important:**
    The bias tensor must be pre-broadcasted to match the output shape.
    QAddOp's verifier will emit an error if lhs and rhs shapes don't match.
    For bias from QLinear, ensure it's shaped as [B, L, OC] not [OC].
  }];
  let constructor = "mlir::allo::createVivadoSeparateBiasPass()";
  let dependentDialects = [
    "mlir::allo::vivado::VivadoDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // ALLO_MLIR_PASSES