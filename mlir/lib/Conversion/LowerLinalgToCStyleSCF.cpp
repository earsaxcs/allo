/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Conversion/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

/// Local folding pattern for AffineApplyOp.
/// This mirrors the small, local canonicalization used by upstream
/// linalg-to-loops lowering, keeping the resulting IR simpler before
/// lowering affine to SCF.
struct FoldTrivialAffineApplyOp : public RewritePattern {
  FoldTrivialAffineApplyOp(MLIRContext *context)
      : RewritePattern(affine::AffineApplyOp::getOperationName(), 0, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto apply = dyn_cast<affine::AffineApplyOp>(op);
    if (!apply)
      return failure();

    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return failure();

    AffineExpr expr = map.getResult(0);

    // Case 1: constant expression.
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, cst.getValue());
      return success();
    }

    // Case 2/3: identity on a single operand.
    if ((isa<AffineDimExpr>(expr) || isa<AffineSymbolExpr>(expr)) &&
        apply.getNumOperands() == 1) {
      rewriter.replaceOp(op, apply.getOperand(0));
      return success();
    }

    return failure();
  }
};

/// Rewrite any linalg op (buffer semantics) to affine loops and erase it.
struct LowerLinalgToAffineLoopsPattern : public RewritePattern {
  LowerLinalgToAffineLoopsPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return failure();

    if (!linalgOp.hasPureBufferSemantics())
      return rewriter.notifyMatchFailure(
          op, "expected linalg op with pure buffer semantics (bufferized)");

    if (failed(linalg::linalgOpToAffineLoops(rewriter, linalgOp)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

static LogicalResult verifyAllLinalgAreBufferized(Operation *root) {
  LogicalResult status = success();
  root->walk([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (!linalgOp.hasPureBufferSemantics()) {
        op->emitError(
            "linalg lowering requires pure buffer semantics; run bufferization "
            "before allo-lower-linalg-to-cstyle-scf");
        status = failure();
      }
    }
  });
  return status;
}

} // namespace

namespace mlir {
namespace allo {

bool applyLowerLinalgToCStyleSCF(ModuleOp &module) {
  MLIRContext *context = module.getContext();

  if (failed(verifyAllLinalgAreBufferized(module)))
    return false;

  // 1) linalg -> affine loops via greedy rewrite.
  {
    RewritePatternSet patterns(context);
    patterns.add<LowerLinalgToAffineLoopsPattern>(context);

    // Canonicalizations that help keep generated affine simpler.
    memref::DimOp::getCanonicalizationPatterns(patterns, context);
    tensor::DimOp::getCanonicalizationPatterns(patterns, context);
    affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
    patterns.add<FoldTrivialAffineApplyOp>(context);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      return false;
  }

  // 2) affine cleanup + affine -> scf/arith + canonicalization.
  {
    PassManager pm(context);
    pm.addNestedPass<func::FuncOp>(
        mlir::affine::createSimplifyAffineStructuresPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (failed(pm.run(module)))
      return false;
  }

  return true;
}

namespace {
struct LowerLinalgToCStyleSCFPass
    : public LowerLinalgToCStyleSCFBase<LowerLinalgToCStyleSCFPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!applyLowerLinalgToCStyleSCF(module))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerLinalgToCStyleSCFPass() {
  return std::make_unique<LowerLinalgToCStyleSCFPass>();
}

} // namespace allo
} // namespace mlir
