/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AlloSplitForwardBatchLoopPass
 *
 * If module attribute `allo.batch` > 1, split top-level `forward` into:
 * - an inner private function operating on batch=1 shaped memrefs
 * - an outer wrapper that preserves original batch shape and iterates over B
 *   using scf.for + memref.subview + func.call.
 */

#include "PassDetail.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

static constexpr StringLiteral kInnerSplitAttr = "allo.batch.split_forward";
static constexpr StringLiteral kOuterSplitAttr = "allo.batch.split_forward.wrapper";

static bool shouldRewriteMemRefBatchToOne(MemRefType memrefTy, int64_t batch) {
  if (memrefTy.getRank() < 1)
    return false;
  int64_t d0 = memrefTy.getShape()[0];
  return d0 == batch || d0 == ShapedType::kDynamic;
}

static Type rewriteBatchToOne(Type ty, int64_t batch) {
  auto memrefTy = dyn_cast<MemRefType>(ty);
  if (!memrefTy || !shouldRewriteMemRefBatchToOne(memrefTy, batch))
    return ty;

  SmallVector<int64_t> newShape(memrefTy.getShape().begin(),
                                memrefTy.getShape().end());
  newShape[0] = 1;
  return MemRefType::get(newShape, memrefTy.getElementType(),
                         memrefTy.getLayout(), memrefTy.getMemorySpace());
}

static void rewireUsesAllowTypeChange(Value from, Value to) {
  SmallVector<OpOperand *> uses;
  uses.reserve(std::distance(from.use_begin(), from.use_end()));
  for (OpOperand &use : from.getUses())
    uses.push_back(&use);
  for (OpOperand *use : uses)
    use->set(to);
}

static void rewireUsesAllowTypeChange(ValueRange from, ValueRange to) {
  if (from.size() != to.size())
    return;
  for (size_t i = 0; i < from.size(); ++i)
    rewireUsesAllowTypeChange(from[i], to[i]);
}

static FailureOr<Value> buildBatchSubview(OpBuilder &b, Location loc, Value src,
                                          Value iv, int64_t batch) {
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy)
    return src;

  if (!(srcTy.getRank() >= 1 && srcTy.hasStaticShape() &&
        srcTy.getShape()[0] == batch))
    return src;

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  offsets.reserve(srcTy.getRank());
  sizes.reserve(srcTy.getRank());
  strides.reserve(srcTy.getRank());

  offsets.push_back(iv);
  sizes.push_back(b.getIndexAttr(1));
  strides.push_back(b.getIndexAttr(1));

  for (int64_t i = 1; i < srcTy.getRank(); ++i) {
    offsets.push_back(b.getIndexAttr(0));
    sizes.push_back(b.getIndexAttr(srcTy.getShape()[i]));
    strides.push_back(b.getIndexAttr(1));
  }

  Type inferred = memref::SubViewOp::inferResultType(srcTy, offsets, sizes, strides);
  auto subviewTy = dyn_cast<MemRefType>(inferred);
  if (!subviewTy)
    return failure();

  return b.create<memref::SubViewOp>(loc, subviewTy, src, offsets, sizes, strides)
      .getResult();
}

static void rewriteSpecialTokenExpandFuncsToBatchOne(ModuleOp module,
                                                     int64_t batch) {
  llvm::errs() << "[allo-split-forward] enter rewriteSpecialTokenExpandFuncsToBatchOne\n";
  llvm::errs().flush();
  auto rewriteOne = [&](StringRef name) {
    auto fn = module.lookupSymbol<func::FuncOp>(name);
    if (!fn || fn.isExternal() || !fn.getBody().hasOneBlock())
      return;
    llvm::errs() << "[allo-split-forward] rewriting special func: " << name << "\n";
    llvm::errs().flush();

    auto oldType = fn.getFunctionType();
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;
    newInputs.reserve(oldType.getNumInputs());
    newResults.reserve(oldType.getNumResults());
    for (Type t : oldType.getInputs())
      newInputs.push_back(rewriteBatchToOne(t, batch));
    for (Type t : oldType.getResults())
      newResults.push_back(rewriteBatchToOne(t, batch));

    fn.setType(FunctionType::get(module.getContext(), newInputs, newResults));
    for (unsigned i = 0; i < fn.getNumArguments(); ++i)
      fn.getArgument(i).setType(newInputs[i]);

    SmallVector<memref::AllocOp> allocs;
    fn.walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });
    for (memref::AllocOp alloc : allocs) {
      auto oldTy = alloc.getType();
      if (oldTy.getRank() < 1 || oldTy.getShape()[0] != batch)
        continue;
      auto newTy = cast<MemRefType>(rewriteBatchToOne(oldTy, batch));
      OpBuilder b(alloc);
      auto newAlloc = b.create<memref::AllocOp>(alloc.getLoc(), newTy,
                                                alloc.getDynamicSizes());
      newAlloc->setAttrs(alloc->getAttrs());
      rewireUsesAllowTypeChange(alloc.getResult(), newAlloc.getResult());
      alloc.erase();
    }

    SmallVector<affine::AffineForOp> loops;
    fn.walk([&](affine::AffineForOp forOp) { loops.push_back(forOp); });
    for (affine::AffineForOp forOp : loops) {
      if (forOp.hasConstantLowerBound() && forOp.hasConstantUpperBound() &&
          forOp.getConstantLowerBound() == 0 &&
          forOp.getConstantUpperBound() == batch) {
        forOp.setConstantUpperBound(1);
      }
    }
  };

  rewriteOne("ViTTokenExpand_vit_embeddings_expand1");
  rewriteOne("ViTTokenExpand_vit_embeddings_expand2");
  llvm::errs() << "[allo-split-forward] leave rewriteSpecialTokenExpandFuncsToBatchOne\n";
  llvm::errs().flush();
}

static void rewriteInnerDynamicActivationsToBatchOne(func::FuncOp inner,
                                                     int64_t batch) {
  MLIRContext *ctx = inner.getContext();
  ModuleOp module = inner->getParentOfType<ModuleOp>();

  // NOTE：目前特判function判断，后续考虑替代
  SmallVector<func::CallOp> calls;
  inner.walk([&](func::CallOp call) { calls.push_back(call); });
  for (func::CallOp call : calls) {
    StringRef name = call.getCallee();
    if (name != "ViTTokenExpand_vit_embeddings_expand1" &&
        name != "ViTTokenExpand_vit_embeddings_expand2")
      continue;

    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(call.getNumResults());
    bool changed = false;
    for (Type t : call.getResultTypes()) {
      Type nt = rewriteBatchToOne(t, batch);
      newResultTypes.push_back(nt);
      changed |= (nt != t);
    }
    if (!changed)
      continue;

    OpBuilder b(call);
    auto newCall = b.create<func::CallOp>(call.getLoc(), name, newResultTypes,
                                          call.getOperands());
    newCall->setAttrs(call->getAttrs());
    rewireUsesAllowTypeChange(call.getResults(), newCall.getResults());
    call.erase();
  }

  auto rewriteBatchMemRefConsumers =
      [&](Value oldSrc, Value newSrc) {
        SmallVector<Operation *> users(oldSrc.getUsers().begin(), oldSrc.getUsers().end());
        for (Operation *user : users) {
          if (auto reshape = dyn_cast<memref::ReshapeOp>(user)) {
            if (reshape.getSource() != oldSrc)
              continue;
            auto oldResTy = dyn_cast<MemRefType>(reshape.getResult().getType());
            if (!oldResTy)
              continue;
            auto newResTy = dyn_cast<MemRefType>(rewriteBatchToOne(oldResTy, batch));
            if (!newResTy)
              continue;

            Value newShape = reshape.getShape();
            if (auto getGlobal = newShape.getDefiningOp<memref::GetGlobalOp>()) {
              if (auto globalOp = module.lookupSymbol<memref::GlobalOp>(getGlobal.getName())) {
                if (auto init =
                        dyn_cast_or_null<DenseIntElementsAttr>(globalOp.getInitialValueAttr())) {
                  SmallVector<int64_t> vals;
                  vals.reserve(init.getNumElements());
                  for (APInt v : init.getValues<APInt>())
                    vals.push_back(v.getSExtValue());
                  if (!vals.empty() && vals[0] == batch) {
                    vals[0] = 1;
                    auto oldGlobalTy = globalOp.getType();
                    auto ety = dyn_cast<IntegerType>(oldGlobalTy.getElementType());
                    if (ety) {
                      auto tensorTy = RankedTensorType::get(oldGlobalTy.getShape(), ety);
                      SmallVector<APInt> apVals;
                      apVals.reserve(vals.size());
                      for (int64_t v : vals)
                        apVals.push_back(APInt(ety.getWidth(), v, true));
                      auto newInit = DenseIntElementsAttr::get(tensorTy, apVals);
                      std::string newName = getGlobal.getName().str() + "_batch1";
                      auto newGlobal = module.lookupSymbol<memref::GlobalOp>(newName);
                      if (!newGlobal) {
                        OpBuilder mb(module.getContext());
                        mb.setInsertionPointToStart(module.getBody());
                        newGlobal = mb.create<memref::GlobalOp>(
                            reshape.getLoc(), newName,
                          StringAttr::get(ctx, "private"), oldGlobalTy,
                            newInit, globalOp.getConstant(), globalOp.getAlignmentAttr());
                      }
                      OpBuilder b(reshape);
                      newShape = b.create<memref::GetGlobalOp>(
                          reshape.getLoc(), oldGlobalTy, newGlobal.getName());
                    }
                  }
                }
              }
            }

            OpBuilder b(reshape);
            auto newOp = b.create<memref::ReshapeOp>(reshape.getLoc(), newResTy, newSrc,
                                                     newShape);
            newOp->setAttrs(reshape->getAttrs());
            rewireUsesAllowTypeChange(reshape.getResult(), newOp.getResult());
            reshape.erase();
            continue;
          }

          if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
            if (subview.getSource() != oldSrc)
              continue;
            auto srcTy = dyn_cast<MemRefType>(newSrc.getType());
            if (!srcTy || srcTy.getRank() < 1)
              continue;

            auto normalizeIndexLike = [&](ArrayRef<OpFoldResult> in,
                                          SmallVector<OpFoldResult> &out) {
              out.clear();
              out.reserve(in.size());
              for (OpFoldResult ofr : in) {
                if (auto v = ofr.dyn_cast<Value>()) {
                  out.push_back(v);
                  continue;
                }
                auto attr = ofr.dyn_cast<Attribute>();
                if (!attr)
                  return false;
                auto intAttr = dyn_cast<IntegerAttr>(attr);
                if (!intAttr)
                  return false;
                out.push_back(IntegerAttr::get(IndexType::get(ctx),
                                               intAttr.getValue().getSExtValue()));
              }
              return true;
            };

            SmallVector<OpFoldResult> mixedOffsets;
            SmallVector<OpFoldResult> mixedSizes;
            SmallVector<OpFoldResult> mixedStrides;
            if (!normalizeIndexLike(subview.getMixedOffsets(), mixedOffsets) ||
                !normalizeIndexLike(subview.getMixedSizes(), mixedSizes) ||
                !normalizeIndexLike(subview.getMixedStrides(), mixedStrides)) {
              subview.emitRemark() << "skip subview batch rewrite due to non-index "
                                      "static offset/size/stride";
              continue;
            }
            if (mixedSizes.empty())
              continue;
            mixedSizes[0] = IntegerAttr::get(IndexType::get(ctx), 1);

            Type inferredTy =
                memref::SubViewOp::inferResultType(srcTy, mixedOffsets, mixedSizes,
                                                   mixedStrides);
            auto newResTy = dyn_cast<MemRefType>(inferredTy);
            if (!newResTy)
              continue;

            OpBuilder b(subview);
            auto newOp = b.create<memref::SubViewOp>(subview.getLoc(), newResTy, newSrc,
                                                     mixedOffsets, mixedSizes,
                                                     mixedStrides);
            for (NamedAttribute na : subview->getAttrs()) {
              StringRef key = na.getName().getValue();
              if (key == "operandSegmentSizes" || key == "static_offsets" ||
                  key == "static_sizes" || key == "static_strides")
                continue;
              newOp->setAttr(na.getName(), na.getValue());
            }
            rewireUsesAllowTypeChange(subview.getResult(), newOp.getResult());
            subview.erase();
            continue;
          }

          if (auto cast = dyn_cast<memref::CastOp>(user)) {
            if (cast.getSource() != oldSrc)
              continue;
            auto oldResTy = dyn_cast<MemRefType>(cast.getResult().getType());
            if (!oldResTy)
              continue;
            auto newResTy = dyn_cast<MemRefType>(rewriteBatchToOne(oldResTy, batch));
            if (!newResTy)
              continue;
            OpBuilder b(cast);
            auto newOp = b.create<memref::CastOp>(cast.getLoc(), newResTy, newSrc);
            newOp->setAttrs(cast->getAttrs());
            rewireUsesAllowTypeChange(cast.getResult(), newOp.getResult());
            cast.erase();
            continue;
          }

          if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(user)) {
            if (reinterpretCast.getSource() != oldSrc)
              continue;
            auto oldResTy = dyn_cast<MemRefType>(reinterpretCast.getResult().getType());
            if (!oldResTy)
              continue;
            auto newResTy = dyn_cast<MemRefType>(rewriteBatchToOne(oldResTy, batch));
            if (!newResTy)
              continue;
            OpBuilder b(reinterpretCast);
            auto mixedOffsets = reinterpretCast.getMixedOffsets();
            if (mixedOffsets.empty())
              continue;
            auto newOp = b.create<memref::ReinterpretCastOp>(
                reinterpretCast.getLoc(), newResTy, newSrc,
                mixedOffsets.front(), reinterpretCast.getMixedSizes(),
                reinterpretCast.getMixedStrides());
            newOp->setAttrs(reinterpretCast->getAttrs());
            rewireUsesAllowTypeChange(reinterpretCast.getResult(),
                                      newOp.getResult());
            reinterpretCast.erase();
            continue;
          }

          if (auto collapse = dyn_cast<memref::CollapseShapeOp>(user)) {
            if (collapse.getSrc() != oldSrc)
              continue;
            auto oldResTy = dyn_cast<MemRefType>(collapse.getResultType());
            if (!oldResTy)
              continue;
            auto newResTy = dyn_cast<MemRefType>(rewriteBatchToOne(oldResTy, batch));
            if (!newResTy)
              continue;
            OpBuilder b(collapse);
            auto newOp = b.create<memref::CollapseShapeOp>(
                collapse.getLoc(), newResTy, newSrc,
                collapse.getReassociationIndices());
            newOp->setAttrs(collapse->getAttrs());
            rewireUsesAllowTypeChange(collapse.getResult(), newOp.getResult());
            collapse.erase();
            continue;
          }

          if (auto expand = dyn_cast<memref::ExpandShapeOp>(user)) {
            if (expand.getSrc() != oldSrc)
              continue;
            auto oldResTy = dyn_cast<MemRefType>(expand.getResultType());
            if (!oldResTy)
              continue;
            auto newResTy = dyn_cast<MemRefType>(rewriteBatchToOne(oldResTy, batch));
            if (!newResTy)
              continue;
            OpBuilder b(expand);
            auto newOp = b.create<memref::ExpandShapeOp>(
                expand.getLoc(), newResTy, newSrc,
                expand.getReassociationIndices());
            newOp->setAttrs(expand->getAttrs());
            rewireUsesAllowTypeChange(expand.getResult(), newOp.getResult());
            expand.erase();
            continue;
          }
        }
      };

  SmallVector<memref::AllocOp> allocs;
  inner.walk([&](memref::AllocOp op) { allocs.push_back(op); });

  for (memref::AllocOp alloc : allocs) {
    auto oldType = alloc.getType();
    // Rewrite only tensors whose leading extent is exactly `batch`.
    if (oldType.getRank() == 0 || oldType.getShape()[0] == ShapedType::kDynamic ||
        oldType.getShape()[0] != batch)
      continue;

    auto newType = cast<MemRefType>(rewriteBatchToOne(oldType, batch));
    SmallVector<Value> dynSizes(alloc.getDynamicSizes().begin(),
                                alloc.getDynamicSizes().end());

    OpBuilder b(alloc);
    memref::AllocOp newAlloc;
    if (auto align = alloc.getAlignment()) {
      newAlloc = b.create<memref::AllocOp>(alloc.getLoc(), newType, dynSizes,
                                           IntegerAttr::get(
                                               IntegerType::get(ctx, 64),
                                               static_cast<int64_t>(*align)));
    } else {
      newAlloc = b.create<memref::AllocOp>(alloc.getLoc(), newType, dynSizes);
    }

    rewriteBatchMemRefConsumers(alloc.getResult(), newAlloc.getResult());
    rewireUsesAllowTypeChange(alloc.getResult(), newAlloc.getResult());
    alloc.erase();
  }

  // Final safeguard: if a subview source is batch-1, force sizes[0] to 1.
  SmallVector<memref::SubViewOp> subviews;
  inner.walk([&](memref::SubViewOp op) { subviews.push_back(op); });
  for (memref::SubViewOp subview : subviews) {
    auto srcTy = dyn_cast<MemRefType>(subview.getSource().getType());
    if (!srcTy || srcTy.getRank() < 1 || !srcTy.hasStaticShape() ||
        srcTy.getShape()[0] != 1)
      continue;

    auto normalizeIndexLike = [&](ArrayRef<OpFoldResult> in,
                                  SmallVector<OpFoldResult> &out) {
      out.clear();
      out.reserve(in.size());
      for (OpFoldResult ofr : in) {
        if (auto v = ofr.dyn_cast<Value>()) {
          out.push_back(v);
          continue;
        }
        auto attr = ofr.dyn_cast<Attribute>();
        if (!attr)
          return false;
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (!intAttr)
          return false;
        out.push_back(IntegerAttr::get(IndexType::get(ctx),
                                       intAttr.getValue().getSExtValue()));
      }
      return true;
    };

    SmallVector<OpFoldResult> mixedOffsets;
    SmallVector<OpFoldResult> mixedSizes;
    SmallVector<OpFoldResult> mixedStrides;
    if (!normalizeIndexLike(subview.getMixedOffsets(), mixedOffsets) ||
        !normalizeIndexLike(subview.getMixedSizes(), mixedSizes) ||
        !normalizeIndexLike(subview.getMixedStrides(), mixedStrides))
      continue;
    if (mixedSizes.empty())
      continue;

    mixedSizes[0] = IntegerAttr::get(IndexType::get(ctx), 1);
    auto inferredTy = memref::SubViewOp::inferResultType(srcTy, mixedOffsets,
                                                         mixedSizes,
                                                         mixedStrides);
    auto newResTy = dyn_cast<MemRefType>(inferredTy);
    if (!newResTy)
      continue;

    OpBuilder b(subview);
    auto newSubview = b.create<memref::SubViewOp>(
        subview.getLoc(), newResTy, subview.getSource(), mixedOffsets,
        mixedSizes, mixedStrides);
    for (NamedAttribute na : subview->getAttrs()) {
      StringRef key = na.getName().getValue();
      if (key == "operandSegmentSizes" || key == "static_offsets" ||
          key == "static_sizes" || key == "static_strides")
        continue;
      newSubview->setAttr(na.getName(), na.getValue());
    }
    rewireUsesAllowTypeChange(subview.getResult(), newSubview.getResult());
    subview.erase();
  }

  SmallVector<memref::DimOp> dims;
  inner.walk([&](memref::DimOp op) { dims.push_back(op); });
  for (memref::DimOp dimOp : dims) {
    auto srcTy = dyn_cast<MemRefType>(dimOp.getSource().getType());
    if (!srcTy || srcTy.getRank() < 1)
      continue;

    auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantIndexOp>();
    if (!cst || cst.value() != 0)
      continue;

    if (srcTy.getShape()[0] != 1)
      continue;

    OpBuilder b(dimOp);
    Value one = b.create<arith::ConstantIndexOp>(dimOp.getLoc(), 1);
    dimOp.replaceAllUsesWith(one);
    dimOp.erase();
  }
}

static bool hasStaticBatchLeadingAllocs(func::FuncOp func, int64_t batch) {
  bool found = false;
  func.walk([&](memref::AllocOp alloc) {
    auto ty = alloc.getType();
    if (ty.getRank() < 1)
      return;
    if (ty.getShape()[0] == batch)
      found = true;
  });
  return found;
}

static LogicalResult diagnoseMixedBatchInLinalgOps(func::FuncOp func,
                                                   int64_t batch) {
  bool foundMixed = false;
  func.walk([&](Operation *op) {
    if (!op->getName().getStringRef().starts_with("linalg."))
      return;

    bool hasBatchOne = false;
    bool hasBatchN = false;
    auto checkMemRef = [&](Type ty) {
      auto mt = dyn_cast<MemRefType>(ty);
      if (!mt || mt.getRank() < 1 || !mt.hasStaticShape())
        return;
      int64_t d0 = mt.getShape()[0];
      hasBatchOne |= (d0 == 1);
      hasBatchN |= (d0 == batch);
    };

    for (Value v : op->getOperands())
      checkMemRef(v.getType());
    for (Value v : op->getResults())
      checkMemRef(v.getType());

    if (hasBatchOne && hasBatchN) {
      op->emitError() << "mixed leading batch dims (1 and " << batch
                      << ") in linalg op memref operands/results";
      foundMixed = true;
    }
  });

  return success(!foundMixed);
}

class AlloSplitForwardBatchLoopPass
    : public PassWrapper<AlloSplitForwardBatchLoopPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlloSplitForwardBatchLoopPass)

  StringRef getArgument() const final { return "allo-split-forward-batch-loop"; }

  StringRef getDescription() const final {
    return "Split forward into B=1 inner function and batch-loop wrapper";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::errs() << "[allo-split-forward] runOnOperation begin\n";
    llvm::errs().flush();

    auto batchAttr = module->getAttrOfType<IntegerAttr>("allo.batch");
    if (!batchAttr)
      return;
    int64_t batch = batchAttr.getInt();
    if (batch <= 1)
      return;

    llvm::errs() << "[allo-split-forward] before rewriteSpecial, batch=" << batch << "\n";
    llvm::errs().flush();
    rewriteSpecialTokenExpandFuncsToBatchOne(module, batch);
    llvm::errs() << "[allo-split-forward] after rewriteSpecial\n";
    llvm::errs().flush();

    func::FuncOp forward;
    for (auto f : module.getOps<func::FuncOp>()) {
      if (f.isExternal())
        continue;
      if (f->hasAttr(kInnerSplitAttr) || f->hasAttr(kOuterSplitAttr))
        continue;
      StringRef n = f.getSymName();
      if (n == "forward" || n.starts_with("forward_")) {
        forward = f;
        break;
      }
    }
    if (!forward)
      return;

    if (!forward.getBody().hasOneBlock()) {
      forward.emitError() << "allo-split-forward-batch-loop requires single-block forward";
      signalPassFailure();
      return;
    }

    // Current implementation only safely rewrites dynamic batch-leading allocs.
    // If static batch-leading allocs exist, skip splitting to avoid emitting
    // inconsistent shapes inside the cloned function.
    // if (hasStaticBatchLeadingAllocs(forward, batch)) {
    //   forward.emitWarning() << "Skipping forward batch loop splitting due to static batch-leading allocs";
    //   return;
    // }

    auto oldType = forward.getFunctionType();
    SmallVector<Type> innerInputs;
    SmallVector<Type> innerResults;
    innerInputs.reserve(oldType.getNumInputs());
    innerResults.reserve(oldType.getNumResults());

    for (Type t : oldType.getInputs())
      innerInputs.push_back(rewriteBatchToOne(t, batch));
    for (Type t : oldType.getResults())
      innerResults.push_back(rewriteBatchToOne(t, batch));

    for (Type t : oldType.getResults()) {
      auto m = dyn_cast<MemRefType>(t);
      if (!m || !m.hasStaticShape()) {
        forward.emitError() << "allo-split-forward-batch-loop only supports static memref return types";
        signalPassFailure();
        return;
      }
    }

    OpBuilder moduleBuilder(module.getContext());
    moduleBuilder.setInsertionPoint(forward);

    auto inner = cast<func::FuncOp>(forward->clone());
    SymbolTable::setSymbolName(inner, (forward.getSymName() + "_batch1_impl").str());
    inner.setPrivate();
    moduleBuilder.insert(inner);

    auto innerType = FunctionType::get(module.getContext(), innerInputs, innerResults);
    inner.setType(innerType);

    for (unsigned i = 0; i < inner.getNumArguments(); ++i)
      inner.getArgument(i).setType(innerInputs[i]);

    rewriteInnerDynamicActivationsToBatchOne(inner, batch); // NOTE
    llvm::errs() << "[allo-split-forward] after rewriteInnerDynamicActivationsToBatchOne\n";
    llvm::errs().flush();

    // Emit precise diagnostics for linalg ops that still mix batch=1 and
    // batch=N memref leading dimensions after rewrite.
    if (failed(diagnoseMixedBatchInLinalgOps(inner, batch))) {
      inner.erase();
      signalPassFailure();
      return;
    }

    // Bail out safely if the rewritten inner function is not verifier-clean.
    // This prevents emitting malformed IR for complex models and keeps the
    // original forward untouched.
    if (failed(verify(inner))) {
      inner.erase();
      signalPassFailure();
      return;
    }

    inner->setAttr(kInnerSplitAttr, UnitAttr::get(module.getContext()));
    forward->setAttr(kOuterSplitAttr, UnitAttr::get(module.getContext()));

    Block &outerEntry = forward.getBody().front();
    outerEntry.clear();

    OpBuilder b = OpBuilder::atBlockBegin(&outerEntry);
    Location loc = forward.getLoc();

    SmallVector<Value> outerOutputs;
    outerOutputs.reserve(oldType.getNumResults());
    for (Type t : oldType.getResults()) {
      auto mt = cast<MemRefType>(t);
      outerOutputs.push_back(b.create<memref::AllocOp>(loc, mt).getResult());
    }

    Value lb = b.create<arith::ConstantIndexOp>(loc, 0);
    Value ub = b.create<arith::ConstantIndexOp>(loc, batch);
    Value step = b.create<arith::ConstantIndexOp>(loc, 1);
    auto forOp = b.create<scf::ForOp>(loc, lb, ub, step);

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
    Value iv = forOp.getInductionVar();

    SmallVector<Value> callArgs;
    callArgs.reserve(forward.getNumArguments());
    for (auto [argIdx, arg] : llvm::enumerate(forward.getArguments())) {
      auto subOr = buildBatchSubview(bodyBuilder, loc, arg, iv, batch);
      if (failed(subOr)) {
        forward.emitError() << "failed to create batch subview for argument";
        signalPassFailure();
        return;
      }

      Value callArg = *subOr;
      Type expectedTy = innerInputs[argIdx];
      if (callArg.getType() != expectedTy) {
        auto expectedMemRefTy = dyn_cast<MemRefType>(expectedTy);
        auto gotMemRefTy = dyn_cast<MemRefType>(callArg.getType());
        if (!expectedMemRefTy || !gotMemRefTy) {
          forward.emitError() << "unsupported non-memref argument type mismatch for batch call";
          signalPassFailure();
          return;
        }

        auto tmp = bodyBuilder.create<memref::AllocOp>(loc, expectedMemRefTy);
        bodyBuilder.create<memref::CopyOp>(loc, callArg, tmp.getResult());
        callArg = tmp.getResult();
      }

      callArgs.push_back(callArg);
    }

    auto call = bodyBuilder.create<func::CallOp>(loc, inner.getSymName(), innerResults, callArgs);

    for (unsigned i = 0; i < call.getNumResults(); ++i) {
      auto outOr = buildBatchSubview(bodyBuilder, loc, outerOutputs[i], iv, batch);
      if (failed(outOr)) {
        forward.emitError() << "failed to create batch subview for output";
        signalPassFailure();
        return;
      }
      bodyBuilder.create<memref::CopyOp>(loc, call.getResult(i), *outOr);
    }

    b.setInsertionPointAfter(forOp);
    b.create<func::ReturnOp>(loc, outerOutputs);

  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createAlloSplitForwardBatchLoopPass() {
  return std::make_unique<AlloSplitForwardBatchLoopPass>();
}

} // namespace allo
} // namespace mlir
