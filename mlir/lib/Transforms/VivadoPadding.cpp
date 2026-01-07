/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// VivadoPaddingPass
//
// This pass performs the Vivado/PYNQ padding step:
// - Pad weight/bias globals on the last dimension (zero-filled)
// - Pad activation buffer types (alloc/reshape) on the last dimension *inside*
//   the quantized Vivado region
// - Insert boundary copies using vivado.activation_dynamic_pad right after
//   vivado.quant and right before vivado.dequant
//
// Alignment rule (tileSize is in bytes):
//   padded_last_dim * element_bytes is a multiple of InstrConfig::kTileSize
//
// Notes:
// - This pass does NOT perform any transpose.
// - IntLayerNorm bias_int must NOT be padded.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/PYNQConfig.h"
#include "allo/Dialect/VivadoOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

#define DEBUG_TYPE "vivado-padding"

using namespace mlir;
using namespace mlir::allo;

namespace vivado_ops = mlir::allo::vivado;

namespace {

static int64_t getElementByteWidth(Type elementType) {
  if (auto intTy = dyn_cast<IntegerType>(elementType)) {
    unsigned bits = intTy.getWidth();
    if (bits % 8 == 0)
      return static_cast<int64_t>(bits / 8);
    return 0;
  }
  if (auto floatTy = dyn_cast<FloatType>(elementType)) {
    unsigned bits = floatTy.getWidth();
    if (bits % 8 == 0)
      return static_cast<int64_t>(bits / 8);
    return 0;
  }
  return 0;
}

// Compute minimal element-count alignment such that (nElems * elemBytes) is a
// multiple of tileBytes.
static int64_t getAlignmentInElementsForTileBytes(Type elementType,
                                                  int64_t tileBytes) {
  if (tileBytes <= 0)
    return 1;
  int64_t elemBytes = getElementByteWidth(elementType);
  if (elemBytes <= 0)
    return 1;
  int64_t g = std::gcd(tileBytes, elemBytes);
  int64_t alignElems = tileBytes / g;
  return std::max<int64_t>(alignElems, 1);
}

static int64_t roundUpToMultiple(int64_t value, int64_t multiple) {
  if (value <= 0 || multiple <= 0)
    return value;
  return ((value + multiple - 1) / multiple) * multiple;
}

static SmallVector<int64_t> getPaddedLastDimShape(ArrayRef<int64_t> shape,
                                                  Type elementType,
                                                  int64_t tileBytes) {
  SmallVector<int64_t> padded(shape.begin(), shape.end());
  if (padded.empty())
    return padded;
  int64_t last = padded.back();
  if (last == ShapedType::kDynamic)
    return padded;
  int64_t alignElems = getAlignmentInElementsForTileBytes(elementType, tileBytes);
  padded.back() = roundUpToMultiple(last, alignElems);
  return padded;
}

// Pad the trailing N dimensions (N=1 => last dim; N=2 => last two dims) to a
// multiple of the element-count alignment that matches tileBytes.
static SmallVector<int64_t> getPaddedTrailingDimsShape(ArrayRef<int64_t> shape,
                                                       Type elementType,
                                                       int64_t tileBytes,
                                                       int64_t trailingDims) {
  SmallVector<int64_t> padded(shape.begin(), shape.end());
  if (padded.empty() || trailingDims <= 0)
    return padded;
  int64_t rank = static_cast<int64_t>(padded.size());
  int64_t n = std::min<int64_t>(trailingDims, rank);

  int64_t alignElems = getAlignmentInElementsForTileBytes(elementType, tileBytes);
  for (int64_t i = 0; i < n; ++i) {
    int64_t dimIdx = rank - 1 - i;
    int64_t dim = padded[dimIdx];
    if (dim == ShapedType::kDynamic)
      continue;
    padded[dimIdx] = roundUpToMultiple(dim, alignElems);
  }
  return padded;
}

// Pad an explicit set of dimension indices to the tileBytes element-count
// alignment. This supports padding non-trailing dimensions (e.g. token dim in
// qmatmul rhs_layout variants).
static SmallVector<int64_t>
getPaddedDimsShape(ArrayRef<int64_t> shape, Type elementType, int64_t tileBytes,
                   ArrayRef<int64_t> dimsToPad) {
  SmallVector<int64_t> padded(shape.begin(), shape.end());
  if (padded.empty() || dimsToPad.empty())
    return padded;

  int64_t rank = static_cast<int64_t>(padded.size());
  int64_t alignElems = getAlignmentInElementsForTileBytes(elementType, tileBytes);

  SmallVector<int64_t> dims;
  dims.reserve(dimsToPad.size());
  for (int64_t d : dimsToPad) {
    if (d < 0 || d >= rank)
      continue;
    dims.push_back(d);
  }
  llvm::sort(dims);
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

  for (int64_t dimIdx : dims) {
    int64_t dim = padded[dimIdx];
    if (dim == ShapedType::kDynamic)
      continue;
    padded[dimIdx] = roundUpToMultiple(dim, alignElems);
  }
  return padded;
}

static SmallVector<int64_t>
mapPadDimsThroughReshape(MemRefType srcType, MemRefType dstType,
                         ArrayRef<int64_t> srcPadDims) {
  SmallVector<int64_t> mapped;
  if (!srcType || !dstType)
    return mapped;
  if (!srcType.hasStaticShape() || !dstType.hasStaticShape())
    return mapped;

  ArrayRef<int64_t> srcShape = srcType.getShape();
  ArrayRef<int64_t> dstShape = dstType.getShape();
  int64_t srcRank = srcType.getRank();
  int64_t dstRank = dstType.getRank();
  if (srcRank <= 0 || dstRank <= 0)
    return mapped;

  // Stride in elements for each dim (row-major): stride[i] = prod(shape[i+1..]).
  SmallVector<int64_t> srcStride(srcRank, 1);
  for (int64_t i = srcRank - 2; i >= 0; --i)
    srcStride[i] = srcStride[i + 1] * srcShape[i + 1];

  SmallVector<int64_t> dstStride(dstRank, 1);
  for (int64_t j = dstRank - 2; j >= 0; --j)
    dstStride[j] = dstStride[j + 1] * dstShape[j + 1];

  for (int64_t srcDim : srcPadDims) {
    if (srcDim < 0 || srcDim >= srcRank)
      continue;
    int64_t targetStride = srcStride[srcDim];

    int64_t best = -1;
    int64_t bestNonOne = -1;
    for (int64_t j = 0; j < dstRank; ++j) {
      if (dstStride[j] != targetStride)
        continue;
      if (j > best)
        best = j;
      if (dstShape[j] != 1 && j > bestNonOne)
        bestNonOne = j;
    }

    int64_t chosen = (bestNonOne != -1) ? bestNonOne : best;
    if (chosen != -1)
      mapped.push_back(chosen);
  }

  llvm::sort(mapped);
  mapped.erase(std::unique(mapped.begin(), mapped.end()), mapped.end());
  return mapped;
}

static bool isWeightGlobal(memref::GlobalOp globalOp) {
  auto memrefType = dyn_cast<MemRefType>(globalOp.getType());
  if (!memrefType || memrefType.getRank() != 2)
    return false;
  StringRef name = globalOp.getSymName();
  return name.contains("weight") || name.contains("fc") || name.contains("proj") ||
         name.contains("qkv") || name.contains("query") || name.contains("key") ||
         name.contains("value") || name.contains("dense");
}

static bool is2DBiasGlobal(memref::GlobalOp globalOp) {
  auto memrefType = dyn_cast<MemRefType>(globalOp.getType());
  if (!memrefType || memrefType.getRank() != 2)
    return false;
  StringRef name = globalOp.getSymName();
  return name.contains("bias");
}

// Return true if this global ultimately feeds vivado.int_layernorm's bias_int
// operand (operand #2). Those biases must NOT be padded.
static bool isIntLayerNormBiasGlobal(ModuleOp module, memref::GlobalOp globalOp) {
  StringRef symbolName = globalOp.getSymName();
  llvm::SmallVector<Value, 16> worklist;
  llvm::DenseSet<const void *> visited;

  auto pushIfNew = [&](Value v) {
    if (!v)
      return;
    const void *key = v.getAsOpaquePointer();
    if (visited.contains(key))
      return;
    visited.insert(key);
    worklist.push_back(v);
  };

  module.walk([&](memref::GetGlobalOp getGlobal) {
    if (getGlobal.getName() == symbolName)
      pushIfNew(getGlobal.getResult());
  });

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (!user)
        continue;

      if (auto ln = dyn_cast<vivado_ops::IntLayerNormOp>(user)) {
        if (use.getOperandNumber() == 2)
          return true;
      }

      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        pushIfNew(subview.getResult());
        continue;
      }
      if (auto reshape = dyn_cast<memref::ReshapeOp>(user)) {
        pushIfNew(reshape.getResult());
        continue;
      }
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        pushIfNew(castOp.getResult());
        continue;
      }
      if (auto viewOp = dyn_cast<memref::ViewOp>(user)) {
        pushIfNew(viewOp.getResult());
        continue;
      }
    }
  }

  return false;
}

template <typename T>
static DenseElementsAttr pad2DMatrix(DenseElementsAttr attr,
                                     ArrayRef<int64_t> origShape,
                                     ArrayRef<int64_t> paddedShape) {
  if (origShape.size() != 2 || paddedShape.size() != 2)
    return attr;

  int64_t origRows = origShape[0];
  int64_t origCols = origShape[1];
  int64_t paddedRows = paddedShape[0];
  int64_t paddedCols = paddedShape[1];

  auto values = attr.getValues<T>();
  SmallVector<T> paddedData(paddedRows * paddedCols, static_cast<T>(0));
  for (int64_t i = 0; i < origRows; ++i) {
    for (int64_t j = 0; j < origCols; ++j) {
      paddedData[i * paddedCols + j] = *(values.begin() + i * origCols + j);
    }
  }

  auto newTensorType = RankedTensorType::get(paddedShape, attr.getElementType());
  return DenseElementsAttr::get(newTensorType, llvm::ArrayRef(paddedData));
}

template <typename T>
static DenseElementsAttr pad1DVector(DenseElementsAttr attr,
                                     ArrayRef<int64_t> origShape,
                                     ArrayRef<int64_t> paddedShape) {
  if (origShape.size() != 1 || paddedShape.size() != 1)
    return attr;
  int64_t origN = origShape[0];
  int64_t paddedN = paddedShape[0];
  auto values = attr.getValues<T>();
  SmallVector<T> paddedData(paddedN, static_cast<T>(0));
  for (int64_t i = 0; i < origN; ++i)
    paddedData[i] = *(values.begin() + i);
  auto newTensorType = RankedTensorType::get(paddedShape, attr.getElementType());
  return DenseElementsAttr::get(newTensorType, llvm::ArrayRef(paddedData));
}

static bool isScalarI32ScaleValue(Value v) {
  return v && v.getType().isInteger(32);
}

static bool isVectorI32ScaleValue(Value v) {
  if (!v)
    return false;
  if (auto mr = dyn_cast<MemRefType>(v.getType())) {
    return mr.getRank() == 1 && mr.getElementType().isInteger(32);
  }
  if (auto tt = dyn_cast<RankedTensorType>(v.getType())) {
    return tt.getRank() == 1 && tt.getElementType().isInteger(32);
  }
  return false;
}

static bool isI32ShapedScaleValue(Value v) {
  if (!v)
    return false;
  if (auto mr = dyn_cast<MemRefType>(v.getType()))
    return mr.getRank() >= 1 && mr.getElementType().isInteger(32);
  if (auto tt = dyn_cast<RankedTensorType>(v.getType()))
    return tt.getRank() >= 1 && tt.getElementType().isInteger(32);
  return false;
}

static memref::GlobalOp getBaseGlobalFromValue(ModuleOp module, Value v);

static std::optional<int64_t> getStaticNumElementsFromShapedType(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return std::nullopt;
  if (!shaped.hasStaticShape())
    return std::nullopt;
  return shaped.getNumElements();
}

static std::optional<int64_t> getStaticScaleElementCount(ModuleOp module,
                                                         Value scaleVal) {
  if (!scaleVal)
    return std::nullopt;
  if (auto g = getBaseGlobalFromValue(module, scaleVal)) {
    if (auto mr = dyn_cast<MemRefType>(g.getType())) {
      if (!mr.hasStaticShape())
        return std::nullopt;
      return mr.getNumElements();
    }
  }
  return getStaticNumElementsFromShapedType(scaleVal.getType());
}

static std::optional<int64_t> pickScaleLengthDim(ArrayRef<int64_t> shape) {
  if (shape.empty())
    return std::nullopt;
  if (shape.size() == 1)
    return 0;
  if (shape.size() == 2) {
    // Common conventions: 1xN or Nx1.
    if (shape[0] == 1 && shape[1] != 1)
      return 1;
    if (shape[1] == 1 && shape[0] != 1)
      return 0;
    return 1;
  }
  // Not expected for current scale encodings.
  return std::nullopt;
}

static std::optional<int64_t> getStaticDim(Value shaped, int64_t dimFromEnd) {
  auto mr = dyn_cast<MemRefType>(shaped.getType());
  if (!mr)
    return std::nullopt;
  int64_t rank = mr.getRank();
  if (rank <= 0)
    return std::nullopt;
  int64_t dim = rank + dimFromEnd;
  if (dim < 0 || dim >= rank)
    return std::nullopt;
  int64_t v = mr.getShape()[dim];
  if (v == ShapedType::kDynamic)
    return std::nullopt;
  return v;
}

// Pad a dense memref.global's shape to an explicit newShape with zeros.
// Only supports rank-1 and rank-2 dense initial values.
static LogicalResult padDenseGlobalToExplicitShape(memref::GlobalOp globalOp,
                                                   ArrayRef<int64_t> newShape) {
  auto memrefType = dyn_cast<MemRefType>(globalOp.getType());
  if (!memrefType)
    return success();
  int64_t rank = memrefType.getRank();
  if (rank != 1 && rank != 2)
    return success();
  if (static_cast<int64_t>(newShape.size()) != rank)
    return success();

  ArrayRef<int64_t> oldShape = memrefType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    if (oldShape[i] == ShapedType::kDynamic || newShape[i] == ShapedType::kDynamic)
      return success();
    if (newShape[i] < oldShape[i])
      return success();
  }
  if (llvm::equal(oldShape, newShape))
    return success();

  auto initialValue = globalOp.getInitialValue();
  if (!initialValue)
    return success();
  auto denseAttr = dyn_cast<DenseElementsAttr>(*initialValue);
  if (!denseAttr)
    return success();

  Type elementType = memrefType.getElementType();
  DenseElementsAttr paddedAttr = denseAttr;
  if (rank == 1) {
    if (elementType.isInteger(32)) {
      paddedAttr = pad1DVector<int32_t>(denseAttr, oldShape, newShape);
    } else if (elementType.isInteger(64)) {
      paddedAttr = pad1DVector<int64_t>(denseAttr, oldShape, newShape);
    } else if (elementType.isF32()) {
      paddedAttr = pad1DVector<float>(denseAttr, oldShape, newShape);
    } else {
      return success();
    }
  } else {
    if (elementType.isInteger(8)) {
      paddedAttr = pad2DMatrix<int8_t>(denseAttr, oldShape, newShape);
    } else if (elementType.isInteger(16)) {
      paddedAttr = pad2DMatrix<int16_t>(denseAttr, oldShape, newShape);
    } else if (elementType.isInteger(32)) {
      paddedAttr = pad2DMatrix<int32_t>(denseAttr, oldShape, newShape);
    } else if (elementType.isF32()) {
      paddedAttr = pad2DMatrix<float>(denseAttr, oldShape, newShape);
    } else {
      return success();
    }
  }

  auto newType = MemRefType::get(SmallVector<int64_t>(newShape.begin(), newShape.end()),
                                 elementType, memrefType.getLayout(),
                                 memrefType.getMemorySpace());
  globalOp.setType(newType);
  globalOp.setInitialValueAttr(paddedAttr);

  auto symbolName = globalOp.getSymName();
  globalOp->getParentOfType<ModuleOp>().walk([&](memref::GetGlobalOp getGlobal) {
    if (getGlobal.getName() == symbolName)
      getGlobal.getResult().setType(newType);
  });
  return success();
}

// Pad a dense memref.global's trailing dimension(s) with zeros.
// - rank 1: pads length
// - rank 2: pads rows/cols when trailingDimsToPad=2, or only cols when =1
static LogicalResult padDenseGlobal(memref::GlobalOp globalOp, int64_t tileBytes,
                                   int64_t trailingDimsToPad) {
  auto memrefType = dyn_cast<MemRefType>(globalOp.getType());
  if (!memrefType)
    return success();
  int64_t rank = memrefType.getRank();
  if (rank != 1 && rank != 2)
    return success();

  ArrayRef<int64_t> shape = memrefType.getShape();
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic)
      return success();
  }

  Type elementType = memrefType.getElementType();
  auto paddedShape = getPaddedTrailingDimsShape(shape, elementType, tileBytes,
                                                trailingDimsToPad);
  if (llvm::equal(paddedShape, shape))
    return success();

  auto initialValue = globalOp.getInitialValue();
  if (!initialValue)
    return success();
  auto denseAttr = dyn_cast<DenseElementsAttr>(*initialValue);
  if (!denseAttr)
    return success();

  DenseElementsAttr paddedAttr = denseAttr;
  if (rank == 2) {
    if (elementType.isInteger(8)) {
      paddedAttr = pad2DMatrix<int8_t>(denseAttr, shape, paddedShape);
    } else if (elementType.isInteger(16)) {
      paddedAttr = pad2DMatrix<int16_t>(denseAttr, shape, paddedShape);
    } else if (elementType.isInteger(32)) {
      paddedAttr = pad2DMatrix<int32_t>(denseAttr, shape, paddedShape);
    } else if (elementType.isF32()) {
      paddedAttr = pad2DMatrix<float>(denseAttr, shape, paddedShape);
    } else {
      return success();
    }
  } else {
    if (elementType.isInteger(32)) {
      paddedAttr = pad1DVector<int32_t>(denseAttr, shape, paddedShape);
    } else if (elementType.isInteger(64)) {
      paddedAttr = pad1DVector<int64_t>(denseAttr, shape, paddedShape);
    } else if (elementType.isF32()) {
      paddedAttr = pad1DVector<float>(denseAttr, shape, paddedShape);
    } else {
      return success();
    }
  }

  auto newType = MemRefType::get(paddedShape, elementType, memrefType.getLayout(),
                                 memrefType.getMemorySpace());
  globalOp.setType(newType);
  globalOp.setInitialValueAttr(paddedAttr);

  // Update memref.get_global result types.
  auto symbolName = globalOp.getSymName();
  globalOp->getParentOfType<ModuleOp>().walk([&](memref::GetGlobalOp getGlobal) {
    if (getGlobal.getName() == symbolName)
      getGlobal.getResult().setType(newType);
  });

  return success();
}

static memref::GlobalOp cloneGlobalWithUniqueName(ModuleOp module,
                                                  memref::GlobalOp orig,
                                                  StringRef suffix) {
  std::string base = (orig.getSymName() + suffix).str();
  std::string newName = base;
  int counter = 0;
  while (module.lookupSymbol(newName))
    newName = (base + "_" + std::to_string(++counter));

  Operation *cloned = orig->clone();
  auto newGlobal = cast<memref::GlobalOp>(cloned);
  newGlobal.setSymName(newName);
  module.getBody()->getOperations().insert(std::next(orig->getIterator()),
                                           newGlobal.getOperation());
  return newGlobal;
}

static memref::GlobalOp getOrCreateBoundaryScaleClone(
    ModuleOp module, memref::GlobalOp orig,
    llvm::DenseMap<Operation *, memref::GlobalOp> &cache) {
  auto it = cache.find(orig.getOperation());
  if (it != cache.end())
    return it->second;
  auto cloned = cloneGlobalWithUniqueName(module, orig, "__boundary");
  cache[orig.getOperation()] = cloned;
  return cloned;
}

static memref::GlobalOp getBaseGlobalFromValue(ModuleOp module, Value v) {
  Value cur = v;
  while (cur) {
    if (auto gg = cur.getDefiningOp<memref::GetGlobalOp>()) {
      auto globalOp = module.lookupSymbol<memref::GlobalOp>(gg.getName());
      return globalOp;
    }
    Operation *def = cur.getDefiningOp();
    if (!def)
      break;
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      cur = subview.getSource();
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      cur = castOp.getSource();
      continue;
    }
    if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
      cur = reshape.getSource();
      continue;
    }
    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
      cur = rc.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      cur = view.getSource();
      continue;
    }
    break;
  }
  return memref::GlobalOp();
}

static bool isVivadoQuantRegionMarkerOp(Operation *op) {
  if (!op)
    return false;
  if (op->getName().getDialectNamespace() != "vivado")
    return false;
  if (isa<vivado_ops::QuantOp, vivado_ops::DequantOp>(op))
    return true;
  if (auto b = op->getAttrOfType<BoolAttr>("is_transposed"))
    return b.getValue();
  return false;
}

static bool isVivadoOp(Operation *op) {
  return op && op->getName().getDialectNamespace() == "vivado";
}

static bool isActivationIntMemRef(Value v) {
  auto memrefType = dyn_cast<MemRefType>(v.getType());
  if (!memrefType)
    return false;
  if (!isa<IntegerType>(memrefType.getElementType()))
    return false;
  // Activations are rank 2/3/4; avoid touching 1D scale/bias vectors.
  if (memrefType.getRank() < 2)
    return false;
  return true;
}

static void addIfActivationMemRef(Value v,
                                  llvm::DenseSet<const void *> &visited,
                                  llvm::SmallVectorImpl<Value> &worklist,
                                  llvm::DenseSet<const void *> &padded) {
  if (!isActivationIntMemRef(v))
    return;
  const void *key = v.getAsOpaquePointer();
  if (!visited.contains(key)) {
    visited.insert(key);
    worklist.push_back(v);
  }
  padded.insert(key);
}

// Compute which activation buffers are inside the padded region.
// Seed: outputs of ActivationDynamicPadOp(to_padded=true).
// Propagation:
// - across memref view-like ops (subview/reshape/cast/view/...)
// - across Vivado ops: if an op touches any padded activation buffer, all its
//   activation-int memref operands are considered padded.
static llvm::DenseSet<const void *> computePaddedActivationSet(func::FuncOp func) {
  llvm::DenseSet<const void *> padded;
  llvm::DenseSet<const void *> visited;
  llvm::SmallVector<Value, 32> worklist;

  func.walk([&](vivado_ops::ActivationDynamicPadOp padOp) {
    if (!padOp.getToPadded())
      return;
    // operand0 is the padded output buffer.
    addIfActivationMemRef(padOp.getOutput(), visited, worklist, padded);
  });

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (!user)
        continue;

      // Stop at depad output side: to_padded=false output is unpadded.
      if (auto pad = dyn_cast<vivado_ops::ActivationDynamicPadOp>(user)) {
        bool toPadded = pad.getToPadded();
        if (!toPadded) {
          // cur may be the padded input (operand1). Do not mark operand0.
          continue;
        }
        // For to_padded=true, cur could be output or input depending on
        // traversal; only output is padded. We'll not traverse to input.
        continue;
      }

      // Traverse memref forwarding ops.
      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        addIfActivationMemRef(subview.getResult(), visited, worklist, padded);
        continue;
      }
      if (auto reshape = dyn_cast<memref::ReshapeOp>(user)) {
        addIfActivationMemRef(reshape.getResult(), visited, worklist, padded);
        continue;
      }
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        addIfActivationMemRef(castOp.getResult(), visited, worklist, padded);
        continue;
      }
      if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(user)) {
        addIfActivationMemRef(reinterpretCast.getResult(), visited, worklist, padded);
        continue;
      }
      if (auto viewOp = dyn_cast<memref::ViewOp>(user)) {
        addIfActivationMemRef(viewOp.getResult(), visited, worklist, padded);
        continue;
      }

      // If a Vivado op touches a padded activation buffer, treat all its
      // activation buffers as padded (ensures multi-operand ops like qadd don't
      // mix 197 and 224 shapes).
      if (isVivadoOp(user)) {
        bool touchesPadded = false;
        for (Value operand : user->getOperands()) {
          if (!isActivationIntMemRef(operand))
            continue;
          if (padded.contains(operand.getAsOpaquePointer())) {
            touchesPadded = true;
            break;
          }
        }
        if (touchesPadded) {
          for (Value operand : user->getOperands())
            addIfActivationMemRef(operand, visited, worklist, padded);
        }
        continue;
      }
    }
  }

  return padded;
}

// Compute which activation buffers should be aligned on the last *two*
// dimensions.
//
// Motivation:
// - When transpose mode is enabled (the graph has been toggle-transposed),
//   qlinear/qmatmul output shapes need both trailing dimensions aligned to the
//   same tile rule.
//
// Seed: outputs of QLinear/QMatMul with transpose_mode=true.
// Propagation: same idea as computePaddedActivationSet, but restricted to
// already-padded values.
static llvm::DenseSet<const void *>
computeAlignLastTwoActivationSet(func::FuncOp func,
                                 const llvm::DenseSet<const void *> &paddedSet) {
  llvm::DenseSet<const void *> align2;
  llvm::DenseSet<const void *> visited;
  llvm::SmallVector<Value, 32> worklist;

  auto addIfInPadded = [&](Value v) {
    if (!isActivationIntMemRef(v))
      return;
    const void *key = v.getAsOpaquePointer();
    if (!paddedSet.contains(key))
      return;
    align2.insert(key);
    if (!visited.contains(key)) {
      visited.insert(key);
      worklist.push_back(v);
    }
  };

  func.walk([&](Operation *op) {
    if (!op || op->getName().getDialectNamespace() != "vivado")
      return;

    bool transposeMode = false;
    if (auto b = op->getAttrOfType<BoolAttr>("transpose_mode"))
      transposeMode = b.getValue();
    if (!transposeMode)
      return;

    if (auto ql = dyn_cast<vivado_ops::QLinearOp>(op)) {
      addIfInPadded(ql.getOutput());
      return;
    }
    if (auto qm = dyn_cast<vivado_ops::QMatMulOp>(op)) {
      addIfInPadded(qm.getOutput());
      return;
    }
  });

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (!user)
        continue;

      // Do not propagate across pad boundaries.
      if (auto pad = dyn_cast<vivado_ops::ActivationDynamicPadOp>(user)) {
        continue;
      }

      // Traverse memref forwarding ops.
      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        addIfInPadded(subview.getResult());
        continue;
      }
      if (auto reshape = dyn_cast<memref::ReshapeOp>(user)) {
        addIfInPadded(reshape.getResult());
        continue;
      }
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        addIfInPadded(castOp.getResult());
        continue;
      }
      if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(user)) {
        addIfInPadded(reinterpretCast.getResult());
        continue;
      }
      if (auto viewOp = dyn_cast<memref::ViewOp>(user)) {
        addIfInPadded(viewOp.getResult());
        continue;
      }

      if (isVivadoOp(user)) {
        // If this Vivado op touches any align2 activation buffer, align2 all of
        // its activation-int memref operands (within padded region) to avoid
        // shape divergence.
        bool touchesAlign2 = false;
        for (Value operand : user->getOperands()) {
          if (!isActivationIntMemRef(operand))
            continue;
          if (align2.contains(operand.getAsOpaquePointer())) {
            touchesAlign2 = true;
            break;
          }
        }
        if (touchesAlign2) {
          for (Value operand : user->getOperands())
            addIfInPadded(operand);
        }
        continue;
      }
    }
  }

  return align2;
}

static bool valueUsedInVivadoQuantRegion(Value v) {
  llvm::SmallVector<Value, 16> worklist;
  llvm::DenseSet<const void *> visited;

  auto pushIfNew = [&](Value x) {
    if (!x)
      return;
    const void *key = x.getAsOpaquePointer();
    if (visited.contains(key))
      return;
    visited.insert(key);
    worklist.push_back(x);
  };

  pushIfNew(v);
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (!user)
        continue;

      if (isVivadoQuantRegionMarkerOp(user))
        return true;

      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        pushIfNew(subview.getResult());
        continue;
      }
      if (auto reshape = dyn_cast<memref::ReshapeOp>(user)) {
        pushIfNew(reshape.getResult());
        continue;
      }
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        pushIfNew(castOp.getResult());
        continue;
      }
      if (auto viewOp = dyn_cast<memref::ViewOp>(user)) {
        pushIfNew(viewOp.getResult());
        continue;
      }
    }
  }

  return false;
}

static bool isQuantOutputBuffer(Value buffer) {
  for (OpOperand &use : buffer.getUses()) {
    if (auto q = dyn_cast<vivado_ops::QuantOp>(use.getOwner())) {
      if (use.getOperandNumber() == 0)
        return true;
    }
  }
  return false;
}

static void updateAllocTypeIfNeeded(memref::AllocOp allocOp, int64_t tileBytes,
                                    ArrayRef<int64_t> dimsToPad) {
  auto memrefType = dyn_cast<MemRefType>(allocOp.getType());
  if (!memrefType)
    return;
  if (!isa<IntegerType>(memrefType.getElementType()))
    return;

  auto paddedShape = getPaddedDimsShape(memrefType.getShape(),
                                        memrefType.getElementType(), tileBytes,
                                        dimsToPad);
  SmallVector<int64_t> origShape(memrefType.getShape().begin(), memrefType.getShape().end());
  if (paddedShape == origShape)
    return;

  auto newType = MemRefType::get(paddedShape, memrefType.getElementType(), memrefType.getLayout(),
                                 memrefType.getMemorySpace());
  allocOp.getResult().setType(newType);
}

static void updateReshapeTypeAndShapeGlobalIfNeeded(ModuleOp module,
                                                    memref::ReshapeOp reshapeOp,
                                                    int64_t tileBytes,
                                                    ArrayRef<int64_t> dimsToPad) {
  auto resultType = dyn_cast<MemRefType>(reshapeOp.getResult().getType());
  if (!resultType)
    return;
  if (!isa<IntegerType>(resultType.getElementType()))
    return;

  auto paddedShape = getPaddedDimsShape(resultType.getShape(),
                                        resultType.getElementType(), tileBytes,
                                        dimsToPad);
  SmallVector<int64_t> origShape(resultType.getShape().begin(), resultType.getShape().end());
  if (paddedShape == origShape)
    return;

  auto newType = MemRefType::get(paddedShape, resultType.getElementType(), resultType.getLayout(),
                                 resultType.getMemorySpace());
  reshapeOp.getResult().setType(newType);

  // Update the shape "global" backing the reshape if it is a constant
  // memref.get_global -> memref.global with dense i64 elements.
  Value shapeValue = reshapeOp.getShape();
  auto getGlobal = shapeValue.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return;
  auto globalOp = module.lookupSymbol<memref::GlobalOp>(getGlobal.getName());
  if (!globalOp)
    return;
  auto init = globalOp.getInitialValue();
  if (!init)
    return;
  auto dense = dyn_cast<DenseElementsAttr>(*init);
  if (!dense)
    return;
  if (!dense.getElementType().isInteger(64))
    return;

  SmallVector<int64_t> dims;
  dims.reserve(dense.getNumElements());
  for (auto it : dense.getValues<APInt>())
    dims.push_back(it.getSExtValue());
  if (dims.empty())
    return;

  // Pad the selected dimension(s) in the shape vector.
  Type elemTy = resultType.getElementType();
  int64_t alignElems = getAlignmentInElementsForTileBytes(elemTy, tileBytes);
  int64_t rank = static_cast<int64_t>(dims.size());
  SmallVector<int64_t> idxs;
  idxs.reserve(dimsToPad.size());
  for (int64_t d : dimsToPad) {
    if (d < 0 || d >= rank)
      continue;
    idxs.push_back(d);
  }
  llvm::sort(idxs);
  idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
  for (int64_t idx : idxs)
    dims[idx] = roundUpToMultiple(dims[idx], alignElems);

  SmallVector<APInt> apints;
  apints.reserve(dims.size());
  for (int64_t v : dims)
    apints.push_back(APInt(64, static_cast<uint64_t>(v), true));
  auto newInit = DenseIntElementsAttr::get(cast<RankedTensorType>(dense.getType()), apints);
  globalOp.setInitialValueAttr(newInit);
}

static void insertActivationPadBoundaries(func::FuncOp func, int64_t tileBytes) {
  MLIRContext *ctx = func.getContext();
  OpBuilder builder(ctx);

  // Insert pad right after quant.
  func.walk([&](vivado_ops::QuantOp quantOp) {
    Value unpaddedInt = quantOp.getOutput();
    auto unpaddedType = dyn_cast<MemRefType>(unpaddedInt.getType());
    if (!unpaddedType)
      return;
    if (!isa<IntegerType>(unpaddedType.getElementType()))
      return;
    if (unpaddedType.getRank() < 2)
      return;

    auto paddedShape = getPaddedLastDimShape(unpaddedType.getShape(), unpaddedType.getElementType(), tileBytes);
    SmallVector<int64_t> origShape(unpaddedType.getShape().begin(), unpaddedType.getShape().end());
    if (paddedShape == origShape)
      return;

    auto paddedType = MemRefType::get(paddedShape, unpaddedType.getElementType(),
                                      unpaddedType.getLayout(), unpaddedType.getMemorySpace());

    builder.setInsertionPointAfter(quantOp);
    auto paddedAlloc = builder.create<memref::AllocOp>(quantOp.getLoc(), paddedType);

    auto padOp = builder.create<vivado_ops::ActivationDynamicPadOp>(
        quantOp.getLoc(), paddedAlloc.getResult(), unpaddedInt,
        builder.getBoolAttr(true));

    Value paddedInt = padOp.getOutput();

    // Replace *all* uses of quant output (except the pad op itself) so the
    // padded region truly consumes paddedInt.
    SmallVector<OpOperand *, 16> toReplace;
    for (OpOperand &use : unpaddedInt.getUses()) {
      Operation *user = use.getOwner();
      if (!user || user == padOp.getOperation())
        continue;
      // Do NOT rewrite the quant op's own output-buffer operand; that would
      // make quant write into a buffer defined after it (dominance violation).
      if (user == quantOp.getOperation())
        continue;
      toReplace.push_back(&use);
    }
    for (OpOperand *use : toReplace)
      use->set(paddedInt);
  });

  // Insert depad right before dequant.
  func.walk([&](vivado_ops::DequantOp dequantOp) {
    Value paddedInt = dequantOp.getInput();
    auto paddedType = dyn_cast<MemRefType>(paddedInt.getType());
    if (!paddedType)
      return;
    if (!isa<IntegerType>(paddedType.getElementType()))
      return;

    Value floatOut = dequantOp.getOutput();
    auto floatOutType = dyn_cast<MemRefType>(floatOut.getType());
    if (!floatOutType)
      return;

    // Use dequant output shape as the unpadded shape boundary.
    SmallVector<int64_t> unpaddedShape(floatOutType.getShape().begin(),
                                       floatOutType.getShape().end());
    auto unpaddedIntType = MemRefType::get(unpaddedShape, paddedType.getElementType(),
                                           paddedType.getLayout(), paddedType.getMemorySpace());

    builder.setInsertionPoint(dequantOp);
    auto unpaddedAlloc = builder.create<memref::AllocOp>(dequantOp.getLoc(), unpaddedIntType);
    auto depadOp = builder.create<vivado_ops::ActivationDynamicPadOp>(
        dequantOp.getLoc(), unpaddedAlloc.getResult(), paddedInt,
        builder.getBoolAttr(false));

    dequantOp.getInputMutable().set(depadOp.getOutput());
  });
}

static void insertVitGetFirstTokenReturnSubview(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  OpBuilder builder(ctx);

  // Collect ViT get-first-token outputs that run in transpose mode.
  llvm::SmallVector<std::pair<Value, bool>, 8> firstTokenOutputs;
  func.walk([&](vivado_ops::ViTGetFirstTokenOp op) {
    if (!op.getTransposeMode())
      return;
    bool isTransposed = op.getIsTransposed();
    firstTokenOutputs.push_back({op.getOutput(), isTransposed});
  });
  if (firstTokenOutputs.empty())
    return;

  auto isAliasOf = [](Value v, Value target) -> bool {
    Value cur = v;
    while (cur) {
      if (cur == target)
        return true;
      Operation *def = cur.getDefiningOp();
      if (!def)
        return false;
      if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
        cur = subview.getSource();
        continue;
      }
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        cur = castOp.getSource();
        continue;
      }
      if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
        cur = reshape.getSource();
        continue;
      }
      if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
        cur = rc.getSource();
        continue;
      }
      if (auto view = dyn_cast<memref::ViewOp>(def)) {
        cur = view.getSource();
        continue;
      }
      return false;
    }
    return false;
  };

  func.walk([&](func::ReturnOp ret) {
    if (ret.getNumOperands() == 0)
      return;

    builder.setInsertionPoint(ret);
    for (OpOperand &retOperand : ret->getOpOperands()) {
      Value retVal = retOperand.get();
      auto retType = dyn_cast<MemRefType>(retVal.getType());
      if (!retType)
        continue;
      int64_t rank = retType.getRank();
      if (rank < 2)
        continue;

      bool matched = false;
      bool isTransposed = false;
      for (auto &it : firstTokenOutputs) {
        if (isAliasOf(retVal, it.first)) {
          matched = true;
          isTransposed = it.second;
          break;
        }
      }
      if (!matched)
        continue;

      // Determine which dimension corresponds to token length.
      // In transpose mode, layout is typically (B,D,L) when is_transposed=true;
      // otherwise (B,L,D). We slice token dimension to length 1.
      int64_t tokenDim = isTransposed ? (rank - 1) : 1;
      if (tokenDim < 0 || tokenDim >= rank)
        continue;

      ArrayRef<int64_t> shape = retType.getShape();
      if (shape[tokenDim] == 1)
        continue;

      SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      sizes.reserve(rank);
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

      for (int64_t i = 0; i < rank; ++i) {
        if (i == tokenDim) {
          sizes.push_back(builder.getIndexAttr(1));
          continue;
        }
        if (shape[i] == ShapedType::kDynamic) {
          sizes.push_back(builder.create<memref::DimOp>(ret.getLoc(), retVal, i).getResult());
          continue;
        }
        sizes.push_back(builder.getIndexAttr(shape[i]));
      }

      auto inferredType = llvm::cast<MemRefType>(
          memref::SubViewOp::inferResultType(retType, offsets, sizes, strides));
      Value sliced = builder
                         .create<memref::SubViewOp>(ret.getLoc(), inferredType,
                                                    retVal, offsets, sizes,
                                                    strides)
                         .getResult();
      retOperand.set(sliced);
    }
  });
}

struct VivadoPaddingPass
    : public PassWrapper<VivadoPaddingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VivadoPaddingPass)

  StringRef getArgument() const override { return "vivado-padding"; }
  StringRef getDescription() const override {
    return "Pad weight/bias/activations to hardware tile-size alignment";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vivado::VivadoDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    if (!applyVivadoPadding(module, context))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

bool applyVivadoPadding(ModuleOp &module, MLIRContext *context) {
  (void)context;
  int64_t tileBytes = pynq::InstrConfig::kTileSize;

  LLVM_DEBUG(llvm::dbgs() << "Starting vivado-padding with tileBytes=" << tileBytes << "\n");

  // 1) Pad weight/bias globals (last dim only). Skip IntLayerNorm bias_int.
  module.walk([&](memref::GlobalOp globalOp) {
    if (isWeightGlobal(globalOp)) {
      (void)padDenseGlobal(globalOp, tileBytes, /*trailingDimsToPad=*/1);
      return;
    }
    if (is2DBiasGlobal(globalOp)) {
      if (isIntLayerNormBiasGlobal(module, globalOp))
        return;
      // The global of is_transposed=false QLinear bias may be padded again below.
      (void)padDenseGlobal(globalOp, tileBytes, /*trailingDimsToPad=*/1);
    }
  });

  // 2) Insert boundary dynamic pad ops inside quant region, then compute the
  //    padded interior set and update alloc/reshape types *only* for values in
  //    that set. After activation padding is known, pad vector scales by
  //    *element count* (NOT tileBytes), using special-cased rules.
  llvm::DenseMap<Operation *, memref::GlobalOp> boundaryScaleCloneCache;
  llvm::DenseMap<Operation *, SmallVector<int64_t>> scaleGlobalTargetShapes;
  llvm::DenseSet<Operation *> qlinearBiasGlobalsAlign2ToPad;
  bool allOk = true;

  auto requestScaleGlobalShape = [&](memref::GlobalOp g,
                                     ArrayRef<int64_t> newShape) {
    if (!g)
      return;
    auto mr = dyn_cast<MemRefType>(g.getType());
    if (!mr)
      return;
    if (mr.getRank() != static_cast<int64_t>(newShape.size()))
      return;
    auto &slot = scaleGlobalTargetShapes[g.getOperation()];
    if (slot.empty()) {
      slot.assign(newShape.begin(), newShape.end());
      return;
    }
    // If multiple requests exist, take the max per-dim (monotonic padding).
    for (size_t i = 0; i < newShape.size(); ++i)
      slot[i] = std::max<int64_t>(slot[i], newShape[i]);
  };

  auto maybeRequestScaleToLen = [&](Value scaleVal, int64_t targetLen) {
    if (!isI32ShapedScaleValue(scaleVal))
      return;
    auto g = getBaseGlobalFromValue(module, scaleVal);
    if (!g)
      return;
    auto mr = dyn_cast<MemRefType>(g.getType());
    if (!mr || !mr.hasStaticShape())
      return;
    if (!mr.getElementType().isInteger(32))
      return;

    auto nElems = mr.getNumElements();
    // Per-tensor scale => memref with exactly 1 element.
    if (nElems == 1)
      return;

    ArrayRef<int64_t> oldShape = mr.getShape();
    auto dimOpt = pickScaleLengthDim(oldShape);
    if (!dimOpt)
      return;
    int64_t dim = *dimOpt;
    if (dim < 0 || dim >= mr.getRank())
      return;
    if (oldShape[dim] == ShapedType::kDynamic)
      return;
    if (targetLen <= oldShape[dim])
      return;

    SmallVector<int64_t> newShape(oldShape.begin(), oldShape.end());
    newShape[dim] = targetLen;
    requestScaleGlobalShape(g, newShape);
  };

  module.walk([&](func::FuncOp func) {
    // 2.0) Quant/Dequant are outside the padded region: if they share scale
    // globals with interior ops, clone those globals first so we can pad the
    // interior scales without affecting boundaries.
    {
      OpBuilder builder(func.getContext());
      func.walk([&](Operation *op) {
        if (!op)
          return;
        if (!isa<vivado_ops::QuantOp, vivado_ops::DequantOp>(op))
          return;

        // scale operand is #2 for both quant and dequant.
        if (op->getNumOperands() <= 2)
          return;
        Value scaleVal = op->getOperand(2);
        auto gg = scaleVal.getDefiningOp<memref::GetGlobalOp>();
        if (!gg)
          return;
        auto globalOp = module.lookupSymbol<memref::GlobalOp>(gg.getName());
        if (!globalOp)
          return;
        auto memrefType = dyn_cast<MemRefType>(globalOp.getType());
        if (!memrefType)
          return;
        if (!memrefType.getElementType().isInteger(32))
          return;

        auto cloned = getOrCreateBoundaryScaleClone(module, globalOp,
                                                    boundaryScaleCloneCache);

        builder.setInsertionPoint(gg);
        auto newGG = builder.create<memref::GetGlobalOp>(
            gg.getLoc(), cloned.getType(), cloned.getSymName());
        op->setOperand(2, newGG.getResult());
      });
    }

    insertActivationPadBoundaries(func, tileBytes);
    auto paddedSet = computePaddedActivationSet(func);
    if (paddedSet.empty())
      return;

    auto align2Set = computeAlignLastTwoActivationSet(func, paddedSet);

    auto getQMatMulTokenDimFromEnd = [](StringRef rhsLayout)
        -> std::optional<int64_t> {
      // Token dimension varies with rhs_layout.
      if (rhsLayout == "bhld")
        return -2;
      if (rhsLayout == "blhd")
        return -3;
      if (rhsLayout == "bhdl")
        return -1;
      return std::nullopt;
    };

    llvm::DenseMap<const void *, SmallVector<int64_t>> extraPadDims;
    auto addExtraPadDimFromEnd = [&](Value v, int64_t dimFromEnd) {
      if (!v)
        return;
      if (!isActivationIntMemRef(v))
        return;
      const void *key = v.getAsOpaquePointer();
      if (!paddedSet.contains(key))
        return;
      auto mr = dyn_cast<MemRefType>(v.getType());
      if (!mr)
        return;
      int64_t rank = mr.getRank();
      int64_t dim = rank + dimFromEnd;
      if (dim < 0 || dim >= rank)
        return;
      extraPadDims[key].push_back(dim);
    };

    // QMatMul rhs may require padding a non-trailing token dimension depending
    // on rhs_layout.
    func.walk([&](vivado_ops::QMatMulOp op) {
      auto dimFromEnd = getQMatMulTokenDimFromEnd(op.getRhsLayout());
      if (!dimFromEnd)
        return;
      addExtraPadDimFromEnd(op.getRhs(), *dimFromEnd);
    });
    func.walk([&](vivado_ops::QMatMulIsqrtDOp op) {
      auto dimFromEnd = getQMatMulTokenDimFromEnd(op.getRhsLayout());
      if (!dimFromEnd)
        return;
      addExtraPadDimFromEnd(op.getRhs(), *dimFromEnd);
    });

    auto getPadDimsForActivationValue = [&](Value v) -> SmallVector<int64_t> {
      llvm::DenseMap<const void *, SmallVector<int64_t>> cache;

      std::function<SmallVector<int64_t>(Value)> derive = [&](Value cur)
          -> SmallVector<int64_t> {
        SmallVector<int64_t> dims;
        if (!cur)
          return dims;
        if (!isActivationIntMemRef(cur))
          return dims;

        const void *key = cur.getAsOpaquePointer();
        if (!paddedSet.contains(key))
          return dims;
        if (auto it = cache.find(key); it != cache.end())
          return it->second;

        auto curType = dyn_cast<MemRefType>(cur.getType());
        if (!curType)
          return dims;
        int64_t rank = curType.getRank();
        if (rank <= 0)
          return dims;

        Operation *def = cur.getDefiningOp();

        // Propagate through view-like ops.
        if (auto subview = dyn_cast_or_null<memref::SubViewOp>(def)) {
          dims = derive(subview.getSource());
        } else if (auto castOp = dyn_cast_or_null<memref::CastOp>(def)) {
          dims = derive(castOp.getSource());
        } else if (auto rc = dyn_cast_or_null<memref::ReinterpretCastOp>(def)) {
          dims = derive(rc.getSource());
        } else if (auto view = dyn_cast_or_null<memref::ViewOp>(def)) {
          dims = derive(view.getSource());
        } else if (auto reshape = dyn_cast_or_null<memref::ReshapeOp>(def)) {
          Value src = reshape.getSource();
          auto srcType = dyn_cast<MemRefType>(src.getType());
          auto dstType = dyn_cast<MemRefType>(cur.getType());
          auto srcDims = derive(src);
          dims = mapPadDimsThroughReshape(srcType, dstType, srcDims);
        }

        // Base/default policy if we couldn't derive anything more specific.
        if (dims.empty()) {
          // Default: always pad last dim in quant region.
          dims.push_back(rank - 1);
          // NOTE: for reshape/view/..., alignSet will collect them but possibly not manage them here
          // because above applies the propagation logic first
          if (align2Set.contains(key) && rank >= 2)
            dims.push_back(rank - 2);
        }

        // Always pad last dim for any activation inside padded region.
        dims.push_back(rank - 1);

        // Per-value extra pad dims (e.g. qmatmul rhs token dimension).
        if (auto it = extraPadDims.find(key); it != extraPadDims.end())
          dims.append(it->second.begin(), it->second.end());

        // Clamp to current rank and unique.
        SmallVector<int64_t> clamped;
        clamped.reserve(dims.size());
        for (int64_t d : dims) {
          if (d < 0 || d >= rank)
            continue;
          clamped.push_back(d);
        }
        llvm::sort(clamped);
        clamped.erase(std::unique(clamped.begin(), clamped.end()), clamped.end());

        cache[key] = clamped;
        return clamped;
      };

      return derive(v);
    };

    auto getPaddedActivationDimLen = [&](Value v,
                                         int64_t dimFromEnd)
        -> std::optional<int64_t> {
      auto mr = dyn_cast<MemRefType>(v.getType());
      if (!mr)
        return std::nullopt;
      int64_t rank = mr.getRank();
      if (rank <= 0)
        return std::nullopt;
      int64_t dim = rank + dimFromEnd;
      if (dim < 0 || dim >= rank)
        return std::nullopt;

      auto dimsToPad = getPadDimsForActivationValue(v);
      if (dimsToPad.empty())
        return getStaticDim(v, dimFromEnd);

      auto paddedShape = getPaddedDimsShape(mr.getShape(), mr.getElementType(),
                                            tileBytes, dimsToPad);
      int64_t len = paddedShape[dim];
      if (len == ShapedType::kDynamic)
        return std::nullopt;
      return len;
    };

    // 2.2) Scale padding (by element count) for ops inside padded region.
    // Many special cases per user spec.
    func.walk([&](Operation *genericOp) {
      if (!isVivadoOp(genericOp))
        return;
      if (isa<vivado_ops::QuantOp, vivado_ops::DequantOp,
              vivado_ops::ActivationDynamicPadOp>(genericOp))
        return;

      // Only reason about ops that touch padded activations.
      bool touchesPadded = false;
      for (Value operand : genericOp->getOperands()) {
        if (!isActivationIntMemRef(operand))
          continue;
        if (paddedSet.contains(operand.getAsOpaquePointer())) {
          touchesPadded = true;
          break;
        }
      }
      if (!touchesPadded)
        return;

      // QLinear
      if (auto op = dyn_cast<vivado_ops::QLinearOp>(genericOp)) {
        Value input = op.getInput();
        Value output = op.getOutput();

        // Only handle non per-tensor scales (per-tensor is a shaped memref with
        // exactly 1 element).
        bool hasNonPerTensorScale = false;
        for (Value v : {op.getIscl(), op.getOscl(), op.getOsclInv(), op.getFscl()}) {
          auto n = getStaticScaleElementCount(module, v);
          if (n && *n > 1)
            hasNonPerTensorScale = true;
        }
        if (op.getBscl()) {
          auto n = getStaticScaleElementCount(module, op.getBscl());
          if (n && *n > 1)
            hasNonPerTensorScale = true;
        }
        if (!hasNonPerTensorScale)
          return;

        if (!op.getTransposeMode()) {
          op.emitError("TODO: scale padding for qlinear when transpose_mode=false (per-token) is not implemented");
          allOk = false;
          return;
        }

        // input scale aligns to input last dim
        if (auto len = getPaddedActivationDimLen(input, -1))
          maybeRequestScaleToLen(op.getIscl(), *len);

        // output-related scales: choose last dim if is_transposed=true else -2
        int64_t dimFromEnd = op.getIsTransposed() ? -1 : -2;
        auto outLen = getPaddedActivationDimLen(output, dimFromEnd);
        if (outLen) {
          maybeRequestScaleToLen(op.getOscl(), *outLen);
          maybeRequestScaleToLen(op.getOsclInv(), *outLen);
          maybeRequestScaleToLen(op.getFscl(), *outLen);
          if (op.getBscl())
            maybeRequestScaleToLen(op.getBscl(), *outLen);
        }
        return;
      }

      // QMatMul
      if (auto op = dyn_cast<vivado_ops::QMatMulOp>(genericOp)) {
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Value output = op.getOutput();

        // x_scale like qlinear input
        if (auto len = getPaddedActivationDimLen(lhs, -1))
          maybeRequestScaleToLen(op.getXScale(), *len);

        // y_scale depends on rhs_layout
        StringRef layout = op.getRhsLayout();
        std::optional<int64_t> yLen;
        if (layout == "bhld") {
          yLen = getPaddedActivationDimLen(rhs, -2);
        } else if (layout == "blhd") {
          yLen = getPaddedActivationDimLen(rhs, -3);
        } else if (layout == "bhdl") {
          yLen = getPaddedActivationDimLen(rhs, -1);
        } else {
          op.emitError("unsupported rhs_layout for scale padding: ") << layout;
          allOk = false;
          return;
        }
        if (yLen)
          maybeRequestScaleToLen(op.getYScale(), *yLen);

        // o_scale(+inv) aligns like qlinear output rule
        int64_t dimFromEnd = op.getIsTransposed() ? -1 : -2;
        if (auto outLen = getPaddedActivationDimLen(output, dimFromEnd)) {
          maybeRequestScaleToLen(op.getOScale(), *outLen);
          maybeRequestScaleToLen(op.getOScaleInv(), *outLen);
          maybeRequestScaleToLen(op.getFusedScale(), *outLen);
        }
        return;
      }

      // QMatMulIsqrtD
      if (auto op = dyn_cast<vivado_ops::QMatMulIsqrtDOp>(genericOp)) {
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Value output = op.getOutput();

        if (auto len = getPaddedActivationDimLen(lhs, -1))
          maybeRequestScaleToLen(op.getXScale(), *len);

        StringRef layout = op.getRhsLayout();
        std::optional<int64_t> yLen;
        if (layout == "bhld") {
          yLen = getPaddedActivationDimLen(rhs, -2);
        } else if (layout == "blhd") {
          yLen = getPaddedActivationDimLen(rhs, -3);
        } else if (layout == "bhdl") {
          yLen = getPaddedActivationDimLen(rhs, -1);
        } else {
          op.emitError("unsupported rhs_layout for scale padding: ") << layout;
          allOk = false;
          return;
        }
        if (yLen)
          maybeRequestScaleToLen(op.getYScale(), *yLen);

        int64_t dimFromEnd = op.getIsTransposed() ? -1 : -2;
        if (auto outLen = getPaddedActivationDimLen(output, dimFromEnd)) {
          maybeRequestScaleToLen(op.getOScale(), *outLen);
          maybeRequestScaleToLen(op.getOScaleInv(), *outLen);
          maybeRequestScaleToLen(op.getFusedScale(), *outLen);
        }
        return;
      }

      // IntLayerNorm
      if (auto op = dyn_cast<vivado_ops::IntLayerNormOp>(genericOp)) {
        if (!op.getTransposeMode())
          return;

        // Per-tensor => shaped scale with 1 element, ignore. Otherwise, treat as
        // per-channel for layernorm.
        bool hasNonPerTensorScale = false;
        for (Value v : {op.getIscl(), op.getLscl(), op.getBscl(), op.getFscl(),
                        op.getOscl(), op.getOsclInv()}) {
          auto n = getStaticScaleElementCount(module, v);
          if (n && *n > 1)
            hasNonPerTensorScale = true;
        }
        if (!hasNonPerTensorScale)
          return;

        int64_t dimFromEnd = op.getIsTransposed() ? -2 : -1;
        auto outLen = getPaddedActivationDimLen(op.getOutput(), dimFromEnd);
        if (!outLen)
          return;
        maybeRequestScaleToLen(op.getIscl(), *outLen);
        maybeRequestScaleToLen(op.getLscl(), *outLen);
        maybeRequestScaleToLen(op.getBscl(), *outLen);
        maybeRequestScaleToLen(op.getFscl(), *outLen);
        maybeRequestScaleToLen(op.getOscl(), *outLen);
        maybeRequestScaleToLen(op.getOsclInv(), *outLen);
        return;
      }

      // Other ops: if scale is per-token (vector), align by output dim.
      // Rule: transpose_mode=true and is_transposed=true => align to last dim,
      // else align to second last dim.
      bool transposeMode = false;
      bool isTransposed = false;
      if (auto b = genericOp->getAttrOfType<BoolAttr>("transpose_mode"))
        transposeMode = b.getValue();
      if (auto b = genericOp->getAttrOfType<BoolAttr>("is_transposed"))
        isTransposed = b.getValue();
      if (!transposeMode)
        return;

      // Output is operand 0 by convention in this dialect.
      if (genericOp->getNumOperands() == 0)
        return;
      Value out = genericOp->getOperand(0);
      int64_t dimFromEnd = isTransposed ? -1 : -2;
      auto outLen = getPaddedActivationDimLen(out, dimFromEnd);
      if (!outLen)
        return;

      for (Value operand : genericOp->getOperands()) {
        auto n = getStaticScaleElementCount(module, operand);
        if (!n || *n <= 1)
          continue;
        maybeRequestScaleToLen(operand, *outLen);
      }
    });

    // Special: in transpose mode but output not transposed, qlinear bias may
    // conceptually match the output's last-two dims and should be aligned on
    // both dimensions.
    func.walk([&](vivado_ops::QLinearOp op) {
      if (!op.getTransposeMode())
        return;
      if (op.getIsTransposed())
        return;
      Value bias = op.getBias();
      if (!bias)
        return;
      auto g = getBaseGlobalFromValue(module, bias);
      if (!g)
        return;
      auto memrefType = dyn_cast<MemRefType>(g.getType());
      if (!memrefType || memrefType.getRank() != 2)
        return;
      qlinearBiasGlobalsAlign2ToPad.insert(g.getOperation());
    });

    func.walk([&](memref::AllocOp allocOp) {
      const void *key = allocOp.getResult().getAsOpaquePointer();
      if (!paddedSet.contains(key))
        return;
      auto dimsToPad = getPadDimsForActivationValue(allocOp.getResult());
      updateAllocTypeIfNeeded(allocOp, tileBytes, dimsToPad);
    });
    func.walk([&](memref::ReshapeOp reshapeOp) {
      const void *key = reshapeOp.getResult().getAsOpaquePointer();
      if (!paddedSet.contains(key))
        return;
      auto dimsToPad = getPadDimsForActivationValue(reshapeOp.getResult());
      updateReshapeTypeAndShapeGlobalIfNeeded(module, reshapeOp, tileBytes,
                                              dimsToPad);
    });

    // Compensate ViT first-token extraction when token dimension has been
    // aligned/padded in transpose mode.
    insertVitGetFirstTokenReturnSubview(func);
  });

  // 3) Pad qlinear bias globals (special case) on both dims.
  for (Operation *op : qlinearBiasGlobalsAlign2ToPad) {
    auto g = dyn_cast<memref::GlobalOp>(op);
    if (!g)
      continue;
    (void)padDenseGlobal(g, tileBytes, /*trailingDimsToPad=*/2);
  }

  // 4) Apply scale padding requests (scales padded by element count).
  for (auto &it : scaleGlobalTargetShapes) {
    auto g = dyn_cast<memref::GlobalOp>(it.first);
    if (!g)
      continue;
    (void)padDenseGlobalToExplicitShape(g, it.second);
  }

  return allOk;
}

std::unique_ptr<OperationPass<ModuleOp>> createVivadoPaddingPass() {
  return std::make_unique<VivadoPaddingPass>();
}

} // namespace allo
} // namespace mlir
