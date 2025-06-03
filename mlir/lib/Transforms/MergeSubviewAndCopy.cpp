/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Support/Utils.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <map>
#include <memory>
#include <set>

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

struct SubViewCopyPattern : public OpRewritePattern<memref::CopyOp> {
  SubViewCopyPattern(MLIRContext* ctx)
    : mlir::OpRewritePattern<memref::CopyOp>(ctx) {}

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                  PatternRewriter &rewriter) const {
      auto subviewOp = copyOp.getTarget().getDefiningOp<memref::SubViewOp>();
      if (!subviewOp)
        return failure();

      auto allocOp = subviewOp.getSource().getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        return failure();

      if (auto stringAttr = allocOp->getAttr("name").dyn_cast<StringAttr>()) {
        if (stringAttr.getValue().find("cat") == std::string::npos)
          return failure();
      }
      else
        return failure();

      if (!copyOp.getSource().getType().isa<MemRefType>())
        return failure();

      SmallVector<OpFoldResult> offsets = subviewOp.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
      SmallVector<OpFoldResult> strides = subviewOp.getMixedStrides();

      Location loc = copyOp.getLoc();

      // TODO: replace with my custom op
      

      return success();
  }
};

bool applyMergeSubviewAndCopy(ModuleOp &mod, MLIRContext* ctxPtr) {
  RewritePatternSet patternsSet(ctxPtr);
  patternsSet.add<SubViewCopyPattern>(ctxPtr);

  if (applyPatternsAndFoldGreedily(mod, std::move(patternsSet)).failed()) {
      return false;
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloMergeSubviewAndCopyTransformation : public MergeSubviewAndCopyBase<AlloMergeSubviewAndCopyTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    MLIRContext* ctxPtr = &getContext();
    if (!applyMergeSubviewAndCopy(mod, ctxPtr))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createMergeSubviewAndCopyPass() {
    return std::make_unique<AlloMergeSubviewAndCopyTransformation>();
}

} // namespace allo
} // namespace mlir