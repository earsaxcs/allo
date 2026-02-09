/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * HoistGetGlobalPass
 *
 * This pass hoists `memref.get_global` operations to the beginning of the
 * block that contains them, preserving their relative order.
 *
 * Motivation:
 * - Make global constant materialization available early within a block.
 * - Reduce noise for later passes by canonicalizing placement.
 */

#include "PassDetail.h"

#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

class HoistGetGlobalPass
    : public PassWrapper<HoistGetGlobalPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistGetGlobalPass)

  StringRef getArgument() const final { return "hoist-get-global"; }

  StringRef getDescription() const final {
    return "Hoist memref.get_global ops to the top of their containing blocks";
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
      hoistInFunction(funcOp);
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  void hoistInFunction(func::FuncOp funcOp) {
    llvm::DenseMap<Block *, llvm::SmallVector<Operation *, 8>> opsByBlock;
    llvm::SmallVector<Block *, 32> blocksInOrder;
    llvm::SmallPtrSet<Block *, 32> seenBlocks;

    funcOp.walk([&](memref::GetGlobalOp getGlobal) {
      Block *b = getGlobal->getBlock();
      if (seenBlocks.insert(b).second)
        blocksInOrder.push_back(b);
      opsByBlock[b].push_back(getGlobal.getOperation());
    });

    for (Block *b : blocksInOrder) {
      auto it = opsByBlock.find(b);
      if (it == opsByBlock.end())
        continue;

      auto insertPoint = b->begin();
      for (Operation *op : it->second) {
        if (!op || op->getBlock() != b)
          continue;
        op->moveBefore(b, insertPoint);
        insertPoint = ++op->getIterator();
      }
    }
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createHoistGetGlobalPass() {
  return std::make_unique<HoistGetGlobalPass>();
}

} // namespace allo
} // namespace mlir
