/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYNQ Optimize Subview Globals Pass
 *
 * This pass simplifies memref.subview on static globals used as weights/biases.
 * - If subview is an identity (offsets all zero, sizes cover full source,
 *   strides all one), it removes the subview.
 * - Otherwise, it materializes a new memref.global containing the sliced data
 *   and replaces the subview with a direct memref.get_global to the new global.
 *
 * Motivation: avoid non-contiguous subviews on static data at PYNQ level.
 */

#include "PassDetail.h"

#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <string>

using namespace mlir;
using namespace mlir::allo;

namespace {

class PYNQOptimizeSubviewGlobalsPass
    : public PassWrapper<PYNQOptimizeSubviewGlobalsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQOptimizeSubviewGlobalsPass)

  StringRef getArgument() const final {
    return "pynq-optimize-subview-globals";
  }

  StringRef getDescription() const final {
    return "Simplify subview on static weight/bias globals by slicing globals";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;

private:
  memref::GetGlobalOp getGlobalFromValue(Value v) const;
  memref::GlobalOp lookupGlobal(ModuleOp module,
                                memref::GetGlobalOp getGlobal) const;
  bool isWeightOrBiasGlobal(memref::GlobalOp globalOp) const;
  bool isIdentitySubview(memref::SubViewOp subview,
                         MemRefType srcType) const;
  bool tryElideRedundantSubview(memref::SubViewOp subview) const;
  LogicalResult materializeSlicedGlobal(ModuleOp module,
                                        memref::SubViewOp subview,
                                        memref::GlobalOp globalOp);

  int64_t uniqueId = 0;
  llvm::StringMap<memref::GlobalOp> sliceCache;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

memref::GetGlobalOp
PYNQOptimizeSubviewGlobalsPass::getGlobalFromValue(Value v) const {
  if (!v)
    return nullptr;
  if (auto getg = v.getDefiningOp<memref::GetGlobalOp>())
    return getg;
  if (auto cast = v.getDefiningOp<memref::CastOp>())
    return cast.getSource().getDefiningOp<memref::GetGlobalOp>();
  return nullptr;
}

memref::GlobalOp
PYNQOptimizeSubviewGlobalsPass::lookupGlobal(ModuleOp module,
                                             memref::GetGlobalOp getGlobal) const {
  if (!module || !getGlobal)
    return nullptr;
  return module.lookupSymbol<memref::GlobalOp>(getGlobal.getName());
}

static bool isWeightGlobal(memref::GlobalOp globalOp) {
  auto memrefType = globalOp.getType().dyn_cast<MemRefType>();
  if (!memrefType || memrefType.getRank() != 2)
    return false;
  StringRef name = globalOp.getSymName();
  return name.contains("weight") || name.contains("fc") || name.contains("proj") ||
         name.contains("qkv") || name.contains("query") || name.contains("key") ||
         name.contains("value") || name.contains("dense");
}

static bool is2DBiasGlobal(memref::GlobalOp globalOp) {
  auto memrefType = globalOp.getType().dyn_cast<MemRefType>();
  if (!memrefType)
    return false;
  StringRef name = globalOp.getSymName();
  return name.contains("bias");
}

bool PYNQOptimizeSubviewGlobalsPass::isWeightOrBiasGlobal(
    memref::GlobalOp globalOp) const {
  return isWeightGlobal(globalOp) || is2DBiasGlobal(globalOp);
}

static bool hasStaticVec(ArrayRef<int64_t> vals) {
  for (int64_t v : vals) {
    if (v == ShapedType::kDynamic)
      return false;
  }
  return true;
}

// Used by both global-slicing and redundant-subview elimination.
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape);

bool PYNQOptimizeSubviewGlobalsPass::isIdentitySubview(
    memref::SubViewOp subview, MemRefType srcType) const {
  if (!subview || !srcType)
    return false;

  auto dstType = subview.getType().dyn_cast<MemRefType>();
  if (!dstType || dstType != srcType)
    return false;

  auto offsets = subview.getStaticOffsets();
  auto sizes = subview.getStaticSizes();
  auto strides = subview.getStaticStrides();
  if (!hasStaticVec(offsets) || !hasStaticVec(sizes) || !hasStaticVec(strides))
    return false;

  if (static_cast<int64_t>(offsets.size()) != srcType.getRank())
    return false;

  for (int64_t i = 0, e = srcType.getRank(); i < e; ++i) {
    if (offsets[i] != 0)
      return false;
    if (strides[i] != 1)
      return false;
    if (sizes[i] != srcType.getShape()[i])
      return false;
  }
  return true;
}

bool PYNQOptimizeSubviewGlobalsPass::tryElideRedundantSubview(
    memref::SubViewOp subview) const {
  if (!subview)
    return false;

  auto srcType = subview.getSource().getType().dyn_cast<MemRefType>();
  auto dstType = subview.getType().dyn_cast<MemRefType>();
  if (!srcType || !dstType)
    return false;

  // Only handle fully static shapes for now.
  if (!srcType.hasStaticShape() || !dstType.hasStaticShape())
    return false;

  // This optimization is intended to remove redundant *views* on contiguous
  // storage. Be conservative and require identity layout on the source.
  if (!srcType.getLayout().isIdentity())
    return false;

  auto offsets = subview.getStaticOffsets();
  auto sizes = subview.getStaticSizes();
  auto strides = subview.getStaticStrides();
  if (!hasStaticVec(offsets) || !hasStaticVec(sizes) || !hasStaticVec(strides))
    return false;

  int64_t srcRank = srcType.getRank();
  if (static_cast<int64_t>(offsets.size()) != srcRank)
    return false;

  // Redundant if the subview covers the full source with unit strides.
  for (int64_t i = 0; i < srcRank; ++i) {
    if (offsets[i] != 0)
      return false;
    if (strides[i] != 1)
      return false;
    if (sizes[i] != srcType.getShape()[i])
      return false;
  }

  // If the types already match, the subview is an exact no-op.
  if (dstType == srcType) {
    subview.replaceAllUsesWith(subview.getSource());
    subview.erase();
    return true;
  }

  // More aggressive: sometimes the subview is only used to materialize a
  // different but semantically-equivalent memref type (typically layout
  // metadata). If rank/shape/element type match and there is no rank reduction,
  // we can often drop the subview entirely by rewiring its users to the source.
  // This avoids introducing extra ops like memref.reinterpret_cast.
  auto isDefaultMemorySpace = [](Attribute memSpace) -> bool {
    if (!memSpace)
      return true;
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memSpace))
      return intAttr.getInt() == 0;
    return false;
  };
  auto memSpaceEqual = [&](Attribute a, Attribute b) -> bool {
    if (a == b)
      return true;
    return isDefaultMemorySpace(a) && isDefaultMemorySpace(b);
  };

  llvm::SmallBitVector droppedDims = subview.getDroppedDims();
  bool hasRankReduction = droppedDims.any();
  if (!hasRankReduction && srcType.getRank() == dstType.getRank() &&
      srcType.getShape() == dstType.getShape() &&
      srcType.getElementType() == dstType.getElementType() &&
      memSpaceEqual(srcType.getMemorySpace(), dstType.getMemorySpace())) {
    // Only do this when users are known to accept arbitrary memref layouts.
    // Avoid call/return boundaries which require exact signature types.
    for (OpOperand &use : subview.getResult().getUses()) {
      Operation *owner = use.getOwner();
      StringRef opname = owner->getName().getStringRef();
      if (opname.starts_with("func."))
        return false;
      if (!(opname.starts_with("memref.") || opname.starts_with("linalg.")))
        return false;
    }

    Value src = subview.getSource();
    for (OpOperand &use : llvm::make_early_inc_range(subview.getResult().getUses()))
      use.set(src);
    subview.erase();
    return true;
  }

  // Otherwise, materialize an equivalent view via memref.reinterpret_cast.
  // This handles common rank-reducing cases such as:
  //   memref<1x192x224xi8> -> memref<192x224xi8, strided<[224,1]>>
  // where the subview simply drops a unit dimension.
  SmallVector<int64_t> srcShape(srcType.getShape().begin(),
                                srcType.getShape().end());
  SmallVector<int64_t> dstShape(dstType.getShape().begin(),
                                dstType.getShape().end());
  auto srcStrides = computeRowMajorStrides(srcShape);

  droppedDims = subview.getDroppedDims();
  SmallVector<int64_t> dstStrides;
  dstStrides.reserve(dstType.getRank());
  for (int64_t d = 0; d < srcRank; ++d) {
    if (droppedDims.test(d)) {
      // Dropped dimensions must be size-1 for a full-coverage subview.
      if (srcShape[d] != 1)
        return false;
      continue;
    }
    dstStrides.push_back(srcStrides[d]);
  }

  if (static_cast<int64_t>(dstStrides.size()) != dstType.getRank())
    return false;

  OpBuilder rewriter(subview);
  OpFoldResult offset = rewriter.getIndexAttr(0);

  SmallVector<OpFoldResult> mixedSizes;
  mixedSizes.reserve(dstShape.size());
  for (int64_t s : dstShape)
    mixedSizes.push_back(rewriter.getIndexAttr(s));

  SmallVector<OpFoldResult> mixedStrides;
  mixedStrides.reserve(dstStrides.size());
  for (int64_t st : dstStrides)
    mixedStrides.push_back(rewriter.getIndexAttr(st));

  auto ric = rewriter.create<memref::ReinterpretCastOp>(
      subview.getLoc(), dstType, subview.getSource(), offset, mixedSizes,
      mixedStrides);
  subview.replaceAllUsesWith(ric.getResult());
  subview.erase();
  return true;
}

static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  int64_t rank = static_cast<int64_t>(shape.size());
  SmallVector<int64_t> strides(rank, 1);
  for (int64_t i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

LogicalResult PYNQOptimizeSubviewGlobalsPass::materializeSlicedGlobal(
    ModuleOp module, memref::SubViewOp subview, memref::GlobalOp globalOp) {
  auto srcType = globalOp.getType().dyn_cast<MemRefType>();
  auto dstType = subview.getType().dyn_cast<MemRefType>();
  if (!srcType || !dstType)
    return success();

  if (!srcType.hasStaticShape() || !dstType.hasStaticShape())
    return success();

  // Only handle identity layout for now.
  if (!srcType.getLayout().isIdentity())
    return success();

  // The ViT torch pipeline often creates weight globals with an initial value
  // but without the 'constant' flag. In that case getConstantInitValue() is
  // empty, so prefer getInitialValue().
  DenseElementsAttr denseAttr;
  if (auto initVal = globalOp.getInitialValue(); initVal.has_value()) {
    denseAttr = llvm::dyn_cast<DenseElementsAttr>(initVal.value());
  }
  if (!denseAttr) {
    denseAttr = llvm::dyn_cast_or_null<DenseElementsAttr>(
        globalOp.getConstantInitValue());
  }
  if (!denseAttr)
    return success();

  auto offsets = subview.getStaticOffsets();
  auto sizes = subview.getStaticSizes();
  auto strides = subview.getStaticStrides();
  if (!hasStaticVec(offsets) || !hasStaticVec(sizes) || !hasStaticVec(strides))
    return success();

  SmallVector<int64_t> dstShape(dstType.getShape().begin(),
                                dstType.getShape().end());
  auto elemType = srcType.getElementType();

  // Build cache key.
  std::string key = globalOp.getSymName().str() + ":";
  auto appendVec = [&](ArrayRef<int64_t> vec) {
    for (int64_t v : vec) {
      key.append(std::to_string(v));
      key.push_back(',');
    }
  };
  appendVec(offsets);
  key.push_back('|');
  appendVec(sizes);
  key.push_back('|');
  appendVec(strides);

  auto it = sliceCache.find(key);
  if (it != sliceCache.end()) {
    OpBuilder rewriter(subview);
    auto getg = rewriter.create<memref::GetGlobalOp>(
        subview.getLoc(), it->second.getType(), it->second.getName());
    subview.replaceAllUsesWith(getg.getResult());
    subview.erase();
    return success();
  }

  // Extract source values.
  SmallVector<Attribute> srcValues;
  srcValues.reserve(static_cast<size_t>(denseAttr.getNumElements()));
  for (Attribute v : denseAttr.getValues<Attribute>())
    srcValues.push_back(v);

  SmallVector<int64_t> srcShape(srcType.getShape().begin(),
                                srcType.getShape().end());
  auto srcStrides = computeRowMajorStrides(srcShape);
  auto dstStrides = computeRowMajorStrides(dstShape);

  int64_t total = 1;
  for (int64_t d : dstShape)
    total *= d;

  SmallVector<Attribute> slicedValues;
  slicedValues.reserve(static_cast<size_t>(total));

  int64_t srcRank = srcType.getRank();
  int64_t dstRank = static_cast<int64_t>(dstShape.size());

  // Handle rank-reducing subview: build a mapping from source dims -> result dims.
  // droppedDims marks source dimensions that are dropped in the result type.
  llvm::SmallBitVector droppedDims = subview.getDroppedDims();
  SmallVector<int64_t> srcDimToDstDim(srcRank, -1);
  int64_t nextDst = 0;
  for (int64_t d = 0; d < srcRank; ++d) {
    if (droppedDims.test(d))
      continue;
    if (nextDst >= dstRank)
      return success();
    srcDimToDstDim[d] = nextDst++;
  }
  if (nextDst != dstRank)
    return success();

  SmallVector<int64_t> idx(dstRank, 0);

  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t remainder = linear;
    for (int64_t dim = 0; dim < dstRank; ++dim) {
      idx[dim] = remainder / dstStrides[dim];
      remainder = remainder % dstStrides[dim];
    }

    int64_t srcLinear = 0;
    for (int64_t srcDim = 0; srcDim < srcRank; ++srcDim) {
      int64_t dstDim = srcDimToDstDim[srcDim];
      int64_t dstIdx = (dstDim < 0) ? 0 : idx[dstDim];
      int64_t srcIdx = offsets[srcDim] + dstIdx * strides[srcDim];
      if (srcIdx < 0 || srcIdx >= srcShape[srcDim])
        return success();
      srcLinear += srcIdx * srcStrides[srcDim];
    }

    if (srcLinear < 0 || srcLinear >= static_cast<int64_t>(srcValues.size()))
      return success();
    slicedValues.push_back(srcValues[static_cast<size_t>(srcLinear)]);
  }

  auto tensorType = RankedTensorType::get(dstShape, elemType);
  auto newInitAttr = DenseElementsAttr::get(tensorType, slicedValues);

  // Build new global type with identity layout (contiguous).
    MemRefLayoutAttrInterface layout;
    MemRefType newMemrefType =
      MemRefType::get(dstShape, elemType, layout, srcType.getMemorySpace());

  std::string newName = globalOp.getSymName().str() +
                        "_subview_" + std::to_string(uniqueId++);

  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToStart(module.getBody());
    auto newGlobal = moduleBuilder.create<memref::GlobalOp>(
      subview.getLoc(), newName, globalOp.getSymVisibilityAttr(),
      newMemrefType, newInitAttr, /*constant=*/globalOp.getConstant(),
      /*alignment=*/nullptr);

  sliceCache[key] = newGlobal;

  OpBuilder rewriter(subview);
  auto getg = rewriter.create<memref::GetGlobalOp>(
      subview.getLoc(), newMemrefType, newGlobal.getName());

  subview.replaceAllUsesWith(getg.getResult());
  subview.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Entry
//===----------------------------------------------------------------------===//

void PYNQOptimizeSubviewGlobalsPass::runOnOperation() {
  ModuleOp module = getOperation();

  SmallVector<memref::SubViewOp, 32> subviews;
  module.walk([&](memref::SubViewOp subview) { subviews.push_back(subview); });

  for (memref::SubViewOp subview : subviews) {
    // First, drop any redundant subviews regardless of whether the source is a
    // global. This keeps IR simple and avoids unnecessary view ops later.
    if (tryElideRedundantSubview(subview))
      continue;

    Value source = subview.getSource();
    auto getGlobal = getGlobalFromValue(source);
    if (!getGlobal)
      continue;

    auto globalOp = lookupGlobal(module, getGlobal);
    if (!globalOp)
      continue;

    if (!isWeightOrBiasGlobal(globalOp))
      continue;

    auto srcType = globalOp.getType().dyn_cast<MemRefType>();
    if (!srcType)
      continue;

    if (isIdentitySubview(subview, srcType)) {
      subview.replaceAllUsesWith(source);
      subview.erase();
      continue;
    }

    if (failed(materializeSlicedGlobal(module, subview, globalOp))) {
      signalPassFailure();
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace allo {
std::unique_ptr<OperationPass<ModuleOp>>
createPYNQOptimizeSubviewGlobalsPass() {
  return std::make_unique<PYNQOptimizeSubviewGlobalsPass>();
}
} // namespace allo
} // namespace mlir
