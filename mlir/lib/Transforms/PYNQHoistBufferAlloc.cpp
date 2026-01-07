/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * PYNQ Hoist Buffer Allocation Pass
 * 
 * This pass hoists all pynq.buffer_alloc operations to the entry block of
 * their containing function. This transformation:
 * 1. Makes buffer lifetime analysis easier in subsequent passes
 * 2. Ensures all buffers are declared before use
 * 3. Simplifies buffer ID assignment by having a clear declaration region
 * 
 * The pass should run after vivado->pynq lowering but before buffer allocation.
 * 
 * Example transformation:
 * 
 * Before:
 * ```mlir
 * func.func @kernel(%A: memref<128x128xi8>) {
 *   scf.for %i = ... {
 *     %buf = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
 *     pynq.data_transfer %A -> %buf
 *     ...
 *   }
 * }
 * ```
 * 
 * After:
 * ```mlir
 * func.func @kernel(%A: memref<128x128xi8>) {
 *   %buf = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
 *   scf.for %i = ... {
 *     pynq.data_transfer %A -> %buf
 *     ...
 *   }
 * }
 * ```
 */

#include "PassDetail.h"
#include "allo/Transforms/Passes.h"
#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Dialect/PYNQOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

//===----------------------------------------------------------------------===//
// PYNQHoistBufferAllocPass Implementation
//===----------------------------------------------------------------------===//

namespace {

class PYNQHoistBufferAllocPass 
    : public PassWrapper<PYNQHoistBufferAllocPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQHoistBufferAllocPass)

  StringRef getArgument() const final { 
    return "pynq-hoist-buffer-alloc"; 
  }
  
  StringRef getDescription() const final {
    return "Hoist all pynq.buffer_alloc operations to function entry blocks";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override;

private:
  /// Process a single function and hoist its buffer allocations
  LogicalResult hoistBuffersInFunction(func::FuncOp funcOp);
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Entry Point
//===----------------------------------------------------------------------===//

void PYNQHoistBufferAllocPass::runOnOperation() {
  ModuleOp module = getOperation();
  
  // Process all functions in the module
  auto result = module.walk([&](func::FuncOp funcOp) {
    if (failed(hoistBuffersInFunction(funcOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Buffer Hoisting Logic
//===----------------------------------------------------------------------===//

LogicalResult PYNQHoistBufferAllocPass::hoistBuffersInFunction(
    func::FuncOp funcOp) {
  
  // Collect all buffer_alloc operations in the function
  SmallVector<BufferAllocOp, 8> allocOps;
  funcOp.walk([&](BufferAllocOp allocOp) {
    allocOps.push_back(allocOp);
  });
  
  // If no buffer allocations, nothing to do
  if (allocOps.empty()) {
    return success();
  }
  
  // Get the entry block of the function
  Block &entryBlock = funcOp.front();
  
  // Find the insertion point (after function arguments, before first op)
  auto insertPoint = entryBlock.begin();
  
  // Move all buffer_alloc operations to the entry block
  for (BufferAllocOp allocOp : allocOps) {
    // Skip if already in entry block at correct position
    if (allocOp->getBlock() == &entryBlock) {
      // Update insertion point to be after this alloc
      auto opIter = allocOp->getIterator();
      if (++opIter != entryBlock.end()) {
        insertPoint = opIter;
      }
      continue;
    }
    
    // Move the operation to the entry block
    allocOp->moveBefore(&entryBlock, insertPoint);
    
    // Update insertion point for next alloc
    insertPoint = ++allocOp->getIterator();
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//
namespace mlir { 
namespace allo {
std::unique_ptr<OperationPass<ModuleOp>> createPYNQHoistBufferAllocPass() {
  return std::make_unique<PYNQHoistBufferAllocPass>();
}
}
}
