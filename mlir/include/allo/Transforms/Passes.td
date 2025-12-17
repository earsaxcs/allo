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

//===----------------------------------------------------------------------===//
// PYNQ Backend Passes
//===----------------------------------------------------------------------===//

def PYNQHoistBufferAlloc : Pass<"pynq-hoist-buffer-alloc", "func::FuncOp"> {
  let summary = "Hoist pynq.buffer_alloc ops to function entry block";
  let description = [{
    This pass moves all pynq.buffer_alloc operations to the entry block of
    each function. This simplifies subsequent buffer allocation analysis by
    placing all buffer declarations in a single location.
    
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
    
    This pass should run before PYNQBufferAllocation.
  }];
  let constructor = "mlir::allo::createPYNQHoistBufferAllocPass()";
  
  let dependentDialects = [
    "mlir::allo::pynq::PYNQDialect",
    "mlir::func::FuncDialect"
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

#endif // ALLO_MLIR_PASSES