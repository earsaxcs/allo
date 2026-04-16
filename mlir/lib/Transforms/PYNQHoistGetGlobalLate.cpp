/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYNQHoistGetGlobalLatePass
 *
 * This pass is intended to run after `pynq-mid-lower`. It hoists all
 * `memref.get_global` operations to the entry block of each function so
 * global loads happen before main compute logic.
 *
 * It also deduplicates same-symbol get_global ops in the entry block to
 * encourage reuse of a single SSA handle for each global.
 */

#include "PassDetail.h"

#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

class PYNQHoistGetGlobalLatePass
    : public PassWrapper<PYNQHoistGetGlobalLatePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQHoistGetGlobalLatePass)

  StringRef getArgument() const final { return "pynq-hoist-get-global-late"; }

  StringRef getDescription() const final {
    return "Hoist memref.get_global to function entry after pynq-mid-lower";
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

    SmallVector<memref::GetGlobalOp, 64> getGlobals;
    funcOp.walk([&](memref::GetGlobalOp op) { getGlobals.push_back(op); });

    if (getGlobals.empty())
      return success();

    auto insertPoint = entry.begin();
    for (memref::GetGlobalOp op : getGlobals) {
      if (!op || op->getBlock() == nullptr)
        continue;
      op->moveBefore(&entry, insertPoint);
      insertPoint = ++op->getIterator();
    }

    // Deduplicate by global symbol within function entry block.
    llvm::DenseMap<SymbolRefAttr, memref::GetGlobalOp> canonicalByName;
    SmallVector<memref::GetGlobalOp, 32> toErase;

    for (Operation &it : entry.getOperations()) {
      auto gg = dyn_cast<memref::GetGlobalOp>(it);
      if (!gg)
        continue;

      SymbolRefAttr name = gg.getNameAttr();
      auto found = canonicalByName.find(name);
      if (found == canonicalByName.end()) {
        canonicalByName[name] = gg;
        continue;
      }

      memref::GetGlobalOp canonical = found->second;
      if (gg.getType() != canonical.getType()) {
        gg.emitError() << "cannot deduplicate memref.get_global for symbol "
                       << name.getRootReference().getValue()
                       << ": mismatched result types";
        return failure();
      }

      gg.getResult().replaceAllUsesWith(canonical.getResult());
      toErase.push_back(gg);
    }

    for (memref::GetGlobalOp gg : toErase)
      gg.erase();

    return success();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createPYNQHoistGetGlobalLatePass() {
  return std::make_unique<PYNQHoistGetGlobalLatePass>();
}

} // namespace allo
} // namespace mlir
