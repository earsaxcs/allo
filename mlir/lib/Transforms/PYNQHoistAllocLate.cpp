/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYNQHoistAllocLatePass
 *
 * This pass is intended to run after `pynq-hoist-get-global-late`.
 * It hoists all `memref.alloc` operations to the entry block of each function.
 *
 * Hoisted allocs are inserted after any `memref.get_global` ops in the entry
 * block so globals and allocs are clustered at the top of the function.
 */

#include "PassDetail.h"

#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

class PYNQHoistAllocLatePass
    : public PassWrapper<PYNQHoistAllocLatePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQHoistAllocLatePass)

  StringRef getArgument() const final { return "pynq-hoist-alloc-late"; }

  StringRef getDescription() const final {
    return "Hoist memref.alloc to function entry after get_global hoisting";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto result = module.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration())
        return WalkResult::advance();
      if (failed(hoistInFunction(funcOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  LogicalResult hoistInFunction(func::FuncOp funcOp) {
    Block &entry = funcOp.front();

    SmallVector<memref::AllocOp, 64> allocs;
    funcOp.walk([&](memref::AllocOp op) { allocs.push_back(op); });

    if (allocs.empty())
      return success();

    auto insertPoint = entry.begin();
    for (Operation &it : entry.getOperations()) {
      if (!isa<memref::GetGlobalOp>(it))
        continue;
      insertPoint = std::next(it.getIterator());
    }

    for (memref::AllocOp op : allocs) {
      if (!op || op->getBlock() == nullptr)
        continue;
      op->moveBefore(&entry, insertPoint);
      insertPoint = ++op->getIterator();
    }

    return success();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createPYNQHoistAllocLatePass() {
  return std::make_unique<PYNQHoistAllocLatePass>();
}

} // namespace allo
} // namespace mlir
