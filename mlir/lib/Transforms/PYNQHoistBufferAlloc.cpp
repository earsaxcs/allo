/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * PYNQ Hoist Buffer Allocation Pass
 * 
 * This pass hoists allocation-like ops to the entry block of their containing
 * function.
 *
 * It performs two related cleanups:
 * 1) Hoist: move `pynq.buffer_alloc` ops (and tagged spill-slot `memref.alloc`)
 *    to function entry, simplifying later lowering.
 * 2) Dedupe: after physical IDs are assigned, merge multiple `pynq.buffer_alloc`
 *    ops that allocate the same physical buffer ID into a single SSA value.
 *
 * Pipeline note:
 * - If run before ID assignment, it will only hoist.
 * - If run after `pynq-buffer-allocation`, it will hoist and dedupe same-ID
 *   allocations.
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

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

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
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;

private:
  /// Process a single function and hoist its buffer allocations
  LogicalResult hoistBuffersInFunction(func::FuncOp funcOp);

  LogicalResult dedupePhysicalBuffersInEntry(func::FuncOp funcOp);
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

  static constexpr StringLiteral kSpillSlotAttrName =
      "allo.pynq.spill_slot";

  // Collect hoistable ops in a deterministic function order.
  struct OrderedOp {
    Operation *op;
    uint64_t order;
  };
  SmallVector<OrderedOp, 64> hoistOps;
  uint64_t order = 0;
  for (Block &b : funcOp.getBody().getBlocks()) {
    for (Operation &op : b.getOperations()) {
      bool isBufferAlloc = llvm::isa<pynq::BufferAllocOp>(op);
      bool isSpillSlotAlloc =
          llvm::isa<memref::AllocOp>(op) && op.hasAttr(kSpillSlotAttrName);
      if (isBufferAlloc || isSpillSlotAlloc)
        hoistOps.push_back(OrderedOp{&op, order});
      ++order;
    }
  }

  if (hoistOps.empty())
    return success();

  llvm::stable_sort(hoistOps, [](const OrderedOp &a, const OrderedOp &b) {
    return a.order < b.order;
  });

  Block &entryBlock = funcOp.front();
  auto insertPoint = entryBlock.begin();

  for (const OrderedOp &it : hoistOps) {
    Operation *op = it.op;
    if (!op)
      continue;
    op->moveBefore(&entryBlock, insertPoint);
    insertPoint = ++op->getIterator();
  }

  return dedupePhysicalBuffersInEntry(funcOp);
}

LogicalResult PYNQHoistBufferAllocPass::dedupePhysicalBuffersInEntry(
    func::FuncOp funcOp) {
  Block &entryBlock = funcOp.front();

  // Group physical buffer allocs by assigned buffer ID.
  llvm::DenseMap<unsigned, pynq::BufferAllocOp> canonicalById;
  SmallVector<pynq::BufferAllocOp, 32> toErase;

  for (Operation &op : entryBlock.getOperations()) {
    auto allocOp = llvm::dyn_cast<pynq::BufferAllocOp>(op);
    if (!allocOp)
      continue;
    if (allocOp.isVirtual())
      continue;

    unsigned id = allocOp.getBufferId();

    auto it = canonicalById.find(id);
    if (it == canonicalById.end()) {
      canonicalById[id] = allocOp;
      continue;
    }

    pynq::BufferAllocOp canonical = it->second;
    if (allocOp.getBuffer().getType() != canonical.getBuffer().getType()) {
      allocOp.emitError()
          << "cannot deduplicate pynq.buffer_alloc with same buffer_id=" << id
          << ": mismatched buffer types (capacity/element type differ)";
      canonical.emitRemark() << "canonical buffer_alloc for buffer_id=" << id;
      return failure();
    }

    // Prefer keeping role information if the canonical doesn't have it.
    if (!canonical.hasRole() && allocOp.hasRole())
      canonical.setRoleAttr(allocOp.getRoleAttr());

    allocOp.getBuffer().replaceAllUsesWith(canonical.getBuffer());
    toErase.push_back(allocOp);
  }

  for (auto a : toErase)
    a.erase();

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
