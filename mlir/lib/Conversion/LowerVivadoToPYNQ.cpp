/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// LowerVivadoToPYNQ Pass
// This file implements the lowering of Vivado backend quantized operations
// to low-level PYNQ hardware instruction operations.
//
// Transformations:
// - vivado.qmatmul -> pynq.buffer_alloc + pynq.copy + pynq.matmul
// - vivado.qlinear -> pynq.buffer_alloc + pynq.copy + pynq.matmul
// - vivado.qadd -> pynq.qadd
// - vivado.int_gelu -> pynq.gelu
// - vivado.int_softmax -> pynq.softmax
// - vivado.int_layernorm -> pynq.layernorm
// - vivado.qconv2d -> lowered to tiled loops + pynq ops
// - vivado.qmatmul_isqrtd -> pynq.matmul_instr with scaling
//===----------------------------------------------------------------------===//

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/VivadoOps.h"
#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQConfig.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <string>

using namespace mlir;
using namespace allo;

// Explicitly use vivado namespace for source ops
namespace vivado_ops = mlir::allo::vivado;
// Explicitly use pynq namespace for target ops
namespace pynq_ops = mlir::allo::pynq;

namespace mlir {
namespace allo {

namespace {

static void maybeInsertSetMagic(ModuleOp module) {
  auto magicAttr = module->getAttrOfType<IntegerAttr>("allo.hidden_dim");
  if (!magicAttr)
    return;

  auto *ctx = module.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i32MagicAttr = IntegerAttr::get(i32Ty, magicAttr.getInt());

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    // Only inject into the top-level forward.
    StringRef name = func.getSymName();
    if (!(name == "forward" || name.starts_with("forward_")))
      continue;

    Block &entry = func.getBody().front();

    // Avoid double insertion.
    for (Operation &op : entry.getOperations()) {
      if (isa<pynq_ops::SetMagicOp>(op))
        return;
      break; // only check the first op; we always insert at entry start
    }

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&entry);
    builder.create<pynq_ops::SetMagicOp>(func.getLoc(), i32MagicAttr);
    return;
  }
}

static Value stripCasts(Value v) {
  while (auto cast = v.getDefiningOp<memref::CastOp>())
    v = cast.getSource();
  return v;
}

static FailureOr<uint32_t> tryGetSingleI32PackedValue(Value v,
                                                      ModuleOp module) {
  v = stripCasts(v);

  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
      llvm::APInt api = intAttr.getValue();
      if (api.getBitWidth() != 32)
        api = api.zextOrTrunc(32);
      return static_cast<uint32_t>(api.getZExtValue());
    }
  }

  // Handle memref.get_global -> memref.global with DenseElementsAttr.
  if (auto get = v.getDefiningOp<memref::GetGlobalOp>()) {
    auto global = module.lookupSymbol<memref::GlobalOp>(get.getName());
    if (!global)
      return failure();

    auto init = global.getInitialValue();
    if (!init.has_value())
      return failure();

    auto dense = dyn_cast<DenseElementsAttr>(*init);
    if (!dense)
      return failure();

    auto memrefTy = global.getType().dyn_cast<MemRefType>();
    if (!memrefTy || !memrefTy.getElementType().isInteger(32))
      return failure();
    if (!memrefTy.hasStaticShape() || memrefTy.getRank() != 1 ||
        memrefTy.getShape()[0] != 1)
      return failure();

    auto it = dense.getValues<llvm::APInt>().begin();
    if (it == dense.getValues<llvm::APInt>().end())
      return failure();
    llvm::APInt api = *it;
    if (api.getBitWidth() != 32)
      api = api.zextOrTrunc(32);
    return static_cast<uint32_t>(api.getZExtValue());
  }

  return failure();
}

static bool isLen1I32ScaleMemref(Value v) {
  v = stripCasts(v);
  auto ty = v.getType().dyn_cast<MemRefType>();
  if (!ty || !ty.getElementType().isInteger(32))
    return false;
  if (!ty.hasStaticShape() || ty.getRank() != 1)
    return false;
  return ty.getShape()[0] == 1;
}

static FailureOr<Value> broadcastLen1ScaleTo(Value scale,
                                             int64_t targetLen,
                                             StringRef tag,
                                             Operation *anchor,
                                             PatternRewriter &rewriter) {
  if (targetLen <= 0)
    return failure();

  auto module = anchor->getParentOfType<ModuleOp>();
  if (!module)
    return failure();

  scale = stripCasts(scale);

  // If already the right shape, keep it.
  if (auto memrefTy = scale.getType().dyn_cast<MemRefType>()) {
    if (memrefTy.hasStaticShape() && memrefTy.getRank() == 1 &&
        memrefTy.getElementType().isInteger(32) &&
        memrefTy.getShape()[0] == targetLen)
      return scale;
  }

  // Only broadcast scalar i32 scales in this lowering.
  if (!isLen1I32ScaleMemref(scale) && !scale.getType().isInteger(32))
    return scale;

  auto packedOr = tryGetSingleI32PackedValue(scale, module);
  if (failed(packedOr)) {
    anchor->emitOpError("cannot broadcast scale: expected i32 constant or memref.get_global of memref<1xi32> with DenseElementsAttr");
    return failure();
  }
  uint32_t packed = *packedOr;

  auto i32 = rewriter.getI32Type();
  auto memrefTy = MemRefType::get({targetLen}, i32);

  SmallVector<uint32_t> values;
  values.assign(static_cast<size_t>(targetLen), packed);

  auto initAttr = DenseElementsAttr::get(
      RankedTensorType::get(memrefTy.getShape(), i32), llvm::ArrayRef(values));

  static int64_t uniq = 0;
  std::string symName =
      "__pynq_bcast_scale_" + tag.str() + "_" + std::to_string(uniq++);

  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToStart(module.getBody());
  auto global = moduleBuilder.create<memref::GlobalOp>(
      anchor->getLoc(), symName,
      moduleBuilder.getStringAttr("private"), memrefTy, initAttr,
      /*constant=*/true, /*alignment=*/nullptr);

  auto getg = rewriter.create<memref::GetGlobalOp>(
      anchor->getLoc(), memrefTy, global.getName());
  return getg.getResult();
}

static FailureOr<int8_t> composeQuantCmbFromVivadoQuantMode(
    int8_t quantMode, bool transposeMode, bool isTransposed) {
  const uint8_t mode = static_cast<uint8_t>(quantMode);
  const uint8_t asymBit = mode & 0x1u;
  const uint8_t granularity = (mode >> 1) & 0x3u;

  bool enable1D = false;
  bool alongCol = false;

  switch (granularity) {
  case 0b00: // per-tensor
    enable1D = false;
    alongCol = false;
    break;
  case 0b01: { // per-token
    enable1D = true;
    // transpose_mode=true 时：is_transposed=false => token沿行；true => token沿列
    const bool tokenAlongCol = transposeMode ? isTransposed : false;
    alongCol = tokenAlongCol;
    break;
  }
  case 0b10: { // per-channel
    enable1D = true;
    // channel 与 token 方向互补
    const bool tokenAlongCol = transposeMode ? isTransposed : false;
    alongCol = !tokenAlongCol;
    break;
  }
  default:
    return failure();
  }

  uint8_t cmb = asymBit;
  if (enable1D)
    cmb |= 0x2u;
  if (enable1D && alongCol)
    cmb |= 0x4u;

  return static_cast<int8_t>(cmb);
}

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: vivado.qmatmul -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQMatMulToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QMatMulOp> {
  using OpRewritePattern<vivado_ops::QMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QMatMulOp op,
                                 PatternRewriter &rewriter) const override {
    StringRef layerType = op.getLayerType();
    if (layerType != "attn.QK" && layerType != "attn.SV")
      return rewriter.notifyMatchFailure(op, "qmatmul lowering only implemented for layer_type=attn.QK/attn.SV");

    Location loc = op.getLoc();
    Location loweredLoc = FusedLoc::get(
      rewriter.getContext(),
      {loc, NameLoc::get(rewriter.getStringAttr(
            layerType == "attn.SV"
              ? "pynq.lowered_from_vivado.qmatmul.attn_SV"
              : "pynq.lowered_from_vivado.qmatmul.attn_QK"))});

    // Extract operands.
    Value output = op.getOutput();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value fusedScale = op.getFusedScale();

    // Backend tiling parameters.
    int32_t tileM = op.getTileM();
    int32_t tileN = op.getTileN();
    int32_t tileK = op.getTileK();

    bool transposeMode = op.getTransposeMode();
    bool outIsTransposed = op.getIsTransposed();

    // Lowering to PYNQ backend currently only supports transpose_mode=true.
    if (!transposeMode) {
      op.emitOpError(
          "LowerVivadoToPYNQ: vivado.qmatmul requires transpose_mode=true for PYNQ backend");
      return failure();
    }

    auto lhsType = lhs.getType().dyn_cast<ShapedType>();
    auto rhsType = rhs.getType().dyn_cast<ShapedType>();
    auto outType = output.getType().dyn_cast<ShapedType>();
    if (!lhsType || !rhsType || !outType)
      return rewriter.notifyMatchFailure(op, "lhs/rhs/output must be shaped types");

    // We rely on memref.subview and pynq.copy endpoints later, so enforce memrefs.
    if (!lhs.getType().isa<MemRefType>() || !rhs.getType().isa<MemRefType>() ||
        !output.getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(op, "expected lhs/rhs/output to be memrefs");
    }

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !outType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "dynamic shapes not yet supported");
    }

    if (lhsType.getElementType() != rhsType.getElementType() ||
        lhsType.getElementType() != outType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "expected lhs/rhs/output element types to match");
    }
    if (!lhsType.getElementType().isInteger(8))
      return rewriter.notifyMatchFailure(op, "only i8 matmul lowering is supported");

    // PYNQ matmul carries a scale memref binding in IR (not encoded in instr).
    if (!fusedScale || !fusedScale.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected fused_scale to be a memref for PYNQ binding");

    if (tileM < 1 || tileN < 1 || tileK < 1)
      return rewriter.notifyMatchFailure(op, "invalid tile_m/tile_n/tile_k");

    auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t {
      if (b == 0)
        return 0;
      return (a + b - 1) / b;
    };

    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    int64_t outRank = outType.getRank();
    if (lhsRank != 4 || rhsRank != 4 || outRank != 4)
      return rewriter.notifyMatchFailure(op, "attn.QK lowering currently requires rank-4 lhs/rhs/output");

    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();

    const int64_t B = lhsShape[0];
    const int64_t H = lhsShape[1];
    const int64_t batchCount = B;

    if (B <= 0 || H <= 0)
      return rewriter.notifyMatchFailure(op, "expected positive static dims for qmatmul");

    StringRef rhsLayout = op.getRhsLayout();

    //===------------------------------------------------------------------===//
    // attn.QK lowering
    //===------------------------------------------------------------------===//
    if (layerType == "attn.QK") {
      // attn.QK shape interpretation (transpose_mode=true):
      //   lhs layout is assumed "bhdl": lhs[b, h, d, l]
      //     - d: head_dim (reduction K per head)
      //     - l: M axis
      //   rhs layout is required to be "bhdl" so we can collapse (h,d)
      //   output is assumed "b h M N" (or "b h N M" if is_transposed).
      const int64_t D = lhsShape[2];
      const int64_t M = lhsShape[3];
      if (B <= 0 || H <= 0 || D <= 0 || M <= 0)
        return rewriter.notifyMatchFailure(op, "expected positive static dims for attn.QK qmatmul");

      // Broadcast scalar/len-1 fused_scale along token (l) dimension.
      if (auto bcast = broadcastLen1ScaleTo(fusedScale, M, "qmatmul_attn_QK_fscl",
                                           op, rewriter);
          succeeded(bcast)) {
        fusedScale = *bcast;
      } else {
        return failure();
      }

      if (rhsLayout != "bhdl")
        return rewriter.notifyMatchFailure(op, "attn.QK lowering currently requires rhs_layout=\"bhdl\" (for head+head_dim collapse)");

      // rhs[b, h, d, l]
      if (rhsShape[0] != B || rhsShape[1] != H)
        return rewriter.notifyMatchFailure(op, "rhs batch/head dims mismatch for attn.QK");

      // reduce_dim is the true reduction (K) axis on RHS.
      // For attn.QK with rhs_layout="bhdl" (B,H,D,L), the reduction axis is `d` (2).
      int64_t reduceDim = op.getReduceDim();
      if (reduceDim != 2)
        return rewriter.notifyMatchFailure(op, "attn.QK lowering expects reduce_dim to point to 'd' axis (2) for rhs_layout=bhdl");

      // d must be <= tileK (requested upfront check).
      if (D > tileK) {
        op.emitOpError("LowerVivadoToPYNQ(attn.QK): head_dim d (") << D
                       << ") exceeds tile_k (" << tileK << ")";
        return failure();
      }
      // Also reject unusual cases where head_dim itself exceeds tiling lanes.
      if (D > tileM || D > tileN) {
        op.emitOpError("LowerVivadoToPYNQ(attn.QK): head_dim d (") << D
                       << ") exceeds tile_m/tile_n (" << tileM << "/" << tileN
                       << ")";
        return failure();
      }

      // reduce_k is taken from rhs[reduce_dim].
      const int64_t reduceK = rhsShape[reduceDim];
      if (reduceK != D)
        return rewriter.notifyMatchFailure(op, "expected rhs reduce_k (via reduce_dim) to equal lhs head_dim d");

      const int64_t N = rhsShape[3]; // rhs 'l'
      if (N <= 0)
        return rewriter.notifyMatchFailure(op, "expected positive N dimension on rhs");
      // Require seqlen dimension to already fit tile_n (no partial tiling here).
      if (N > tileN)
        return rewriter.notifyMatchFailure(op, "attn.QK requires rhs seqlen (l) to be pre-split to fit tile_n");
      if (M > tileM)
        return rewriter.notifyMatchFailure(op, "attn.QK requires lhs seqlen (l) to be pre-split to fit tile_m");

      // Validate output shape.
      if (outShape[0] != B || outShape[1] != H)
        return rewriter.notifyMatchFailure(op, "output batch/head dims mismatch");
      if (!outIsTransposed) {
        if (outShape[2] != M || outShape[3] != N)
          return rewriter.notifyMatchFailure(op, "output shape mismatch for non-transposed output");
      } else {
        if (outShape[2] != N || outShape[3] != M)
          return rewriter.notifyMatchFailure(op, "output shape mismatch for transposed output");
      }

      // Head packing rule: tileK must be a multiple of d; then a tileK-wide K
      // slice can contain headsPerGroup heads (each contributes d rows).
      if (tileK % static_cast<int32_t>(D) != 0)
        return rewriter.notifyMatchFailure(op, "tile_k must be a multiple of head_dim d for head packing");
      const int64_t headsPerGroup = static_cast<int64_t>(tileK) / D;
      if (headsPerGroup < 1)
        return rewriter.notifyMatchFailure(op, "invalid headsPerGroup computed from tile_k / d");

      // Validate per-tile buffer footprint fits in on-chip capacity.
      int64_t lhsTileBytes = static_cast<int64_t>(tileK) * tileM * 1;
      int64_t rhsTileBytes = static_cast<int64_t>(tileK) * tileN * 1;
      int64_t outTileBytes = static_cast<int64_t>(tileM) * tileN * 1;
      if (lhsTileBytes > pynq::BufferConfig::kDefaultCapacityBytes ||
          rhsTileBytes > pynq::BufferConfig::kDefaultCapacityBytes ||
          outTileBytes > pynq::BufferConfig::kDefaultCapacityBytes) {
        return rewriter.notifyMatchFailure(op, "tile size exceeds buffer capacity (adjust tiling)");
      }

      int64_t numTilesM = ceilDiv(M, tileM);
      int64_t numTilesN = ceilDiv(N, tileN);
      int64_t numHeadGroups = ceilDiv(H, headsPerGroup);

      // Buffer counts:
      // - lhs/rhs: packed by (head_group, d) and tiled over M/N.
      // - out: head is NOT packed (head_dim is reduced away), so allocate per-head.
      int64_t lhsBufCount = std::max<int64_t>(int64_t{1}, batchCount * numHeadGroups * numTilesM);
      int64_t rhsBufCount = std::max<int64_t>(int64_t{1}, batchCount * numHeadGroups * numTilesN);
      int64_t outBufCount = std::max<int64_t>(int64_t{1}, batchCount * H * numTilesM * numTilesN);

      auto i8Type = rewriter.getIntegerType(8);
      auto bufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

      SmallVector<Value, 4> lhsBufs;
      SmallVector<Value, 4> rhsBufs;
      SmallVector<Value, 4> outBufs;
      lhsBufs.reserve(lhsBufCount);
      rhsBufs.reserve(rhsBufCount);
      outBufs.reserve(outBufCount);

      for (int64_t i = 0; i < lhsBufCount; ++i) {
        lhsBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
            loweredLoc, bufferType, rewriter.getStringAttr("input")));
      }
      for (int64_t i = 0; i < rhsBufCount; ++i) {
        rhsBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
            loweredLoc, bufferType, rewriter.getStringAttr("weight")));
      }
      for (int64_t i = 0; i < outBufCount; ++i) {
        outBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
            loweredLoc, bufferType, rewriter.getStringAttr("output")));
      }

      //===------------------------------------------------------------------===//
      // Step 3: Fully unrolled copies + matmul compute (no scf.for)
      //===------------------------------------------------------------------===//

      auto createI32ConstAt = [&](int32_t value, Location l) {
        return rewriter.create<arith::ConstantOp>(
            l, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
      };
      auto createI32Const = [&](int32_t value) { return createI32ConstAt(value, loweredLoc); };

      auto makeTaggedLoc = [&](StringRef tag) -> Location {
        return FusedLoc::get(
            rewriter.getContext(),
            {loweredLoc, NameLoc::get(rewriter.getStringAttr(tag))});
      };

      auto makeStaticOfr = [&](int64_t v) -> OpFoldResult {
        return rewriter.getIndexAttr(v);
      };

      auto makeRankReducedSubview = [&](Value src, ArrayRef<int64_t> resultShape,
                                        ArrayRef<int64_t> offsets,
                                        ArrayRef<int64_t> sizes,
                                        Location l) -> Value {
        auto srcTy = src.getType().cast<MemRefType>();
        int64_t r = srcTy.getRank();
        SmallVector<OpFoldResult, 4> ofrOffsets;
        SmallVector<OpFoldResult, 4> ofrSizes;
        SmallVector<OpFoldResult, 4> ofrStrides;
        ofrOffsets.reserve(r);
        ofrSizes.reserve(r);
        ofrStrides.reserve(r);
        for (int64_t i = 0; i < r; ++i) {
          ofrOffsets.push_back(makeStaticOfr(offsets[i]));
          ofrSizes.push_back(makeStaticOfr(sizes[i]));
          ofrStrides.push_back(makeStaticOfr(1));
        }
        Type inferred = memref::SubViewOp::inferRankReducedResultType(
            resultShape, srcTy, ofrOffsets, ofrSizes, ofrStrides);
        auto resTy = inferred.cast<MemRefType>();
        return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets,
                                                  ofrSizes, ofrStrides);
      };

      // Collapse head and head_dim (H and D) for lhs/rhs to enable packed slicing.
      // lhs/rhs are both [B, H, D, L] under bhdl.
      SmallVector<ReassociationIndices, 4> reassociation;
      reassociation.push_back(ReassociationIndices{0});
      reassociation.push_back(ReassociationIndices{1, 2});
      reassociation.push_back(ReassociationIndices{3});

      Value lhsCollapsed = rewriter.create<memref::CollapseShapeOp>(
        makeTaggedLoc("pynq.qmatmul.attn_QK.lhs_collapse"), lhs, reassociation);
      Value rhsCollapsed = rewriter.create<memref::CollapseShapeOp>(
        makeTaggedLoc("pynq.qmatmul.attn_QK.rhs_collapse"), rhs, reassociation);

      // Matmul instruction tile_count fields follow qlinear: derived from tileM/tileN
      // and InstrConfig::kTileSize.
      // IMPORTANT: These are instruction encoding fields and must reflect the
      // actual per-tile buffer extents (not the configured tileM/tileN). Using
      // tileM/tileN can over-count on boundary tiles.
      int64_t inputTileCountI64 = ceilDiv(M, static_cast<int64_t>(pynq::InstrConfig::kTileSize));
      int64_t weightTileCountI64 = ceilDiv(N, static_cast<int64_t>(pynq::InstrConfig::kTileSize));
      if (inputTileCountI64 < 1 || inputTileCountI64 > pynq::InstrConfig::kMaxTileCount)
        return rewriter.notifyMatchFailure(op, "input_tile_count out of hardware range [1, 8]");
      if (weightTileCountI64 < 1 || weightTileCountI64 > pynq::InstrConfig::kMaxTileCount)
        return rewriter.notifyMatchFailure(op, "weight_tile_count out of hardware range [1, 8]");

      Value inputTileCountVal = createI32Const(static_cast<int32_t>(inputTileCountI64));
      Value weightTileCountVal = createI32Const(static_cast<int32_t>(weightTileCountI64));
      Value reduceKVal = createI32Const(static_cast<int32_t>(reduceK));
      Value headTileAxisVal = createI32Const(0); // per attn.QK rule
      Value enableBiasVal = createI32Const(0);
      Value enableTransposeVal = createI32Const(outIsTransposed ? 1 : 0);

      // 3.1 Copy rhs tiles: per (b, head_group, nTile).
      for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t hg = 0; hg < numHeadGroups; ++hg) {
          int64_t hStart = hg * headsPerGroup;
          int64_t hCount = std::min<int64_t>(headsPerGroup, H - hStart);
          if (hCount < 1)
            continue;
          int64_t kOffset = hStart * D;
          int64_t kSize = hCount * D;
          for (int64_t nTile = 0; nTile < numTilesN; ++nTile) {
            int64_t nStart = nTile * tileN;
            int64_t nLen = std::min<int64_t>(tileN, N - nStart);
            if (nLen < 1)
              continue;

            // rhsCollapsed: [B, H*D, N]
            SmallVector<int64_t, 4> offsets = {b, kOffset, nStart};
            SmallVector<int64_t, 4> sizes = {1, kSize, nLen};
            Location copyWLoc = makeTaggedLoc("pynq.qmatmul.attn_QK.copy_rhs");
            Value rhsSubview = makeRankReducedSubview(
                rhsCollapsed, /*resultShape=*/{kSize, nLen}, offsets, sizes, copyWLoc);

            int64_t rhsLinear = (b * numHeadGroups + hg) * numTilesN + nTile;
            rewriter.create<pynq_ops::CopyOp>(copyWLoc, rhsSubview, rhsBufs[rhsLinear]);
          }
        }
      }

      // 3.2 Copy lhs tiles: per (b, head_group, mTile).
      for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t hg = 0; hg < numHeadGroups; ++hg) {
          int64_t hStart = hg * headsPerGroup;
          int64_t hCount = std::min<int64_t>(headsPerGroup, H - hStart);
          if (hCount < 1)
            continue;
          int64_t kOffset = hStart * D;
          int64_t kSize = hCount * D;
          for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
            int64_t mStart = mTile * tileM;
            int64_t mLen = std::min<int64_t>(tileM, M - mStart);
            if (mLen < 1)
              continue;

            // lhsCollapsed: [B, H*D, M]
            SmallVector<int64_t, 4> offsets = {b, kOffset, mStart};
            SmallVector<int64_t, 4> sizes = {1, kSize, mLen};
            Location copyInLoc = makeTaggedLoc("pynq.qmatmul.attn_QK.copy_lhs");
            Value lhsSubview = makeRankReducedSubview(
                lhsCollapsed, /*resultShape=*/{kSize, mLen}, offsets, sizes, copyInLoc);

            int64_t lhsLinear = (b * numHeadGroups + hg) * numTilesM + mTile;
            rewriter.create<pynq_ops::CopyOp>(copyInLoc, lhsSubview, lhsBufs[lhsLinear]);
          }
        }
      }

      // 3.3 Compute: for each tile, emit one matmul per head inside the packed buffers.
      for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t hg = 0; hg < numHeadGroups; ++hg) {
          int64_t hStart = hg * headsPerGroup;
          int64_t hCount = std::min<int64_t>(headsPerGroup, H - hStart);
          if (hCount < 1)
            continue;
          for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
            int64_t mStart = mTile * tileM;
            int64_t mLen = std::min<int64_t>(tileM, M - mStart);
            if (mLen < 1)
              continue;
            int64_t lhsLinear = (b * numHeadGroups + hg) * numTilesM + mTile;

            for (int64_t nTile = 0; nTile < numTilesN; ++nTile) {
              int64_t nStart = nTile * tileN;
              int64_t nLen = std::min<int64_t>(tileN, N - nStart);
              if (nLen < 1)
                continue;
              int64_t rhsLinear = (b * numHeadGroups + hg) * numTilesN + nTile;

              for (int64_t localH = 0; localH < hCount; ++localH) {
                int64_t globalH = hStart + localH;
                Value headIdxVal = createI32Const(static_cast<int32_t>(localH));

                int64_t outLinear = ((b * H + globalH) * numTilesM + mTile) * numTilesN + nTile;

                std::string tag = ("pynq.qmatmul.attn_QK.compute.b" + std::to_string(b) +
                                   ".hg" + std::to_string(hg) +
                                   ".h" + std::to_string(globalH) +
                                   ".m" + std::to_string(mTile) +
                                   ".n" + std::to_string(nTile));
                Location computeLoc = makeTaggedLoc(tag);

                rewriter.create<pynq_ops::MatMulOp>(
                    computeLoc, fusedScale,
                    lhsBufs[lhsLinear], rhsBufs[rhsLinear], outBufs[outLinear],
                    inputTileCountVal, weightTileCountVal,
                    reduceKVal, headIdxVal, headTileAxisVal,
                    enableBiasVal, enableTransposeVal);
                rewriter.create<pynq_ops::SyncOp>(makeTaggedLoc("pynq.qmatmul.attn_QK.sync"));

                // Copy output tile back.
                SmallVector<int64_t, 4> outOffsets;
                SmallVector<int64_t, 4> outSizes;
                outOffsets.reserve(4);
                outSizes.reserve(4);
                outOffsets.push_back(b);
                outOffsets.push_back(globalH);
                outSizes.push_back(1);
                outSizes.push_back(1);
                if (!outIsTransposed) {
                  outOffsets.push_back(mStart);
                  outOffsets.push_back(nStart);
                  outSizes.push_back(mLen);
                  outSizes.push_back(nLen);
                  Value outSubview = makeRankReducedSubview(
                      output, /*resultShape=*/{mLen, nLen}, outOffsets, outSizes,
                      makeTaggedLoc("pynq.qmatmul.attn_QK.copy_out"));
                  rewriter.create<pynq_ops::CopyOp>(makeTaggedLoc("pynq.qmatmul.attn_QK.copy_out"),
                                                   outBufs[outLinear], outSubview);
                } else {
                  outOffsets.push_back(nStart);
                  outOffsets.push_back(mStart);
                  outSizes.push_back(nLen);
                  outSizes.push_back(mLen);
                  Value outSubview = makeRankReducedSubview(
                      output, /*resultShape=*/{nLen, mLen}, outOffsets, outSizes,
                      makeTaggedLoc("pynq.qmatmul.attn_QK.copy_out"));
                  rewriter.create<pynq_ops::CopyOp>(makeTaggedLoc("pynq.qmatmul.attn_QK.copy_out"),
                                                   outBufs[outLinear], outSubview);
                }
              }
            }
          }
        }
      }

      rewriter.eraseOp(op);
      return success();
    }

    //===------------------------------------------------------------------===//
    // attn.SV lowering
    //===------------------------------------------------------------------===//
    else if (layerType == "attn.SV") {
      // Expected shapes under transpose_mode=true.
      //   lhs: [B, H, Lk, Lq]  (bh(lk)(lq))
      //   rhs: [B, Lk, H, D]   with rhs_layout="blhd"
      //   out: [B, H, D, Lq] when is_transposed=true
      //        [B, H, Lq, D] when is_transposed=false
      if (rhsLayout != "blhd")
        return rewriter.notifyMatchFailure(op, "attn.SV lowering currently requires rhs_layout=\"blhd\"");

      if (lhsShape[2] <= 0 || lhsShape[3] <= 0)
        return rewriter.notifyMatchFailure(op, "expected positive Lk/Lq dims for attn.SV");

      const int64_t K = lhsShape[2]; // Lk (reduce)
      const int64_t M = lhsShape[3]; // Lq (non-reduce)

      // Broadcast scalar/len-1 fused_scale along token (Lq) dimension.
      if (auto bcast = broadcastLen1ScaleTo(fusedScale, M, "qmatmul_attn_SV_fscl",
                                           op, rewriter);
          succeeded(bcast)) {
        fusedScale = *bcast;
      } else {
        return failure();
      }

      // rhs is [B, Lk, H, D]
      if (rhsShape[0] != B)
        return rewriter.notifyMatchFailure(op, "rhs batch dim mismatch for attn.SV");
      if (rhsShape[1] != K)
        return rewriter.notifyMatchFailure(op, "rhs Lk dim mismatch for attn.SV");
      if (rhsShape[2] != H)
        return rewriter.notifyMatchFailure(op, "rhs head dim mismatch for attn.SV");
      const int64_t D = rhsShape[3];
      if (D <= 0)
        return rewriter.notifyMatchFailure(op, "expected positive head_dim D for attn.SV");

      // reduce_dim is the true reduction (K) axis on RHS.
      int64_t reduceDim = op.getReduceDim();
      if (reduceDim != 1)
        return rewriter.notifyMatchFailure(op, "attn.SV lowering expects reduce_dim=1 for rhs_layout=blhd");
      const int64_t reduceK = rhsShape[reduceDim];
      if (reduceK != K)
        return rewriter.notifyMatchFailure(op, "attn.SV expected rhs reduce_k (via reduce_dim) to match lhs Lk");

      // Pre-split constraints: do not tile/slice K or M here.
      if (K > tileK)
        return rewriter.notifyMatchFailure(op, "attn.SV requires Lk (reduce axis) to be pre-split to fit tile_k");
      if (M > tileM)
        return rewriter.notifyMatchFailure(op, "attn.SV requires Lq to be pre-split to fit tile_m");
      if (D > tileM || D > tileN) {
        op.emitOpError("LowerVivadoToPYNQ(attn.SV): head_dim D (") << D
                       << ") exceeds tile_m/tile_n (" << tileM << "/" << tileN << ")";
        return failure();
      }

      // Validate output shape.
      if (outShape[0] != B || outShape[1] != H)
        return rewriter.notifyMatchFailure(op, "output batch/head dims mismatch for attn.SV");
      if (!outIsTransposed) {
        // Non-transposed expected [B,H,Lq,D]
        if (outShape[2] != M || outShape[3] != D)
          return rewriter.notifyMatchFailure(op, "output shape mismatch for non-transposed attn.SV output");
      } else {
        // Transposed expected [B,H,D,Lq]
        if (outShape[2] != D || outShape[3] != M)
          return rewriter.notifyMatchFailure(op, "output shape mismatch for transposed attn.SV output");
      }

      // Pack multiple heads into one RHS/OUT buffer by collapsing (H,D) into a
      // single packed-N axis. Each head consumes D columns.
      if (D > tileN)
        return rewriter.notifyMatchFailure(op, "attn.SV requires head_dim D to fit within tile_n for head packing");
      const int64_t headsPerGroup = std::max<int64_t>(1, static_cast<int64_t>(tileN) / D);
      const int64_t numHeadGroups = ceilDiv(H, headsPerGroup);
      const int64_t numTilesM = ceilDiv(M, tileM);

      // Validate buffer footprints against capacity.
      int64_t lhsTileBytes = static_cast<int64_t>(tileK) * tileM * 1;
      int64_t rhsTileBytes = static_cast<int64_t>(tileK) * tileN * 1;
      int64_t outTileBytes = static_cast<int64_t>(tileN) * tileM * 1;
      if (lhsTileBytes > pynq::BufferConfig::kDefaultCapacityBytes ||
          rhsTileBytes > pynq::BufferConfig::kDefaultCapacityBytes ||
          outTileBytes > pynq::BufferConfig::kDefaultCapacityBytes) {
        return rewriter.notifyMatchFailure(op, "tile size exceeds buffer capacity (adjust tiling)");
      }

      auto i8Type = rewriter.getIntegerType(8);
      auto bufferType = pynq_ops::BufferType::getVirtual(
          rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

      // Buffer counts:
      // - LHS: head is NOT packed; allocate per (b,h,mTile)
      // - RHS: heads are packed; allocate per (b,head_group)
      // - OUT: heads are packed like RHS; allocate per (b,head_group,mTile)
      int64_t lhsBufCount = std::max<int64_t>(int64_t{1}, batchCount * H * numTilesM);
      int64_t rhsBufCount = std::max<int64_t>(int64_t{1}, batchCount * numHeadGroups);
      int64_t outBufCount = std::max<int64_t>(int64_t{1}, batchCount * numHeadGroups * numTilesM);

      SmallVector<Value, 4> lhsBufs;
      SmallVector<Value, 4> rhsBufs;
      SmallVector<Value, 4> outBufs;
      lhsBufs.reserve(lhsBufCount);
      rhsBufs.reserve(rhsBufCount);
      outBufs.reserve(outBufCount);

      for (int64_t i = 0; i < lhsBufCount; ++i)
        lhsBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
            loweredLoc, bufferType, rewriter.getStringAttr("input")));
      for (int64_t i = 0; i < rhsBufCount; ++i)
        rhsBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
            loweredLoc, bufferType, rewriter.getStringAttr("weight")));
      for (int64_t i = 0; i < outBufCount; ++i)
        outBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
            loweredLoc, bufferType, rewriter.getStringAttr("output")));

      auto createI32ConstAt = [&](int32_t value, Location l) {
        return rewriter.create<arith::ConstantOp>(
            l, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
      };
      auto createI32Const = [&](int32_t value) { return createI32ConstAt(value, loweredLoc); };

      auto makeTaggedLoc = [&](StringRef tag) -> Location {
        return FusedLoc::get(
            rewriter.getContext(),
            {loweredLoc, NameLoc::get(rewriter.getStringAttr(tag))});
      };
      auto makeStaticOfr = [&](int64_t v) -> OpFoldResult { return rewriter.getIndexAttr(v); };
      auto makeRankReducedSubview = [&](Value src, ArrayRef<int64_t> resultShape,
                                        ArrayRef<int64_t> offsets,
                                        ArrayRef<int64_t> sizes,
                                        Location l) -> Value {
        auto srcTy = src.getType().cast<MemRefType>();
        int64_t r = srcTy.getRank();
        SmallVector<OpFoldResult, 4> ofrOffsets;
        SmallVector<OpFoldResult, 4> ofrSizes;
        SmallVector<OpFoldResult, 4> ofrStrides;
        ofrOffsets.reserve(r);
        ofrSizes.reserve(r);
        ofrStrides.reserve(r);
        for (int64_t i = 0; i < r; ++i) {
          ofrOffsets.push_back(makeStaticOfr(offsets[i]));
          ofrSizes.push_back(makeStaticOfr(sizes[i]));
          ofrStrides.push_back(makeStaticOfr(1));
        }
        Type inferred = memref::SubViewOp::inferRankReducedResultType(
            resultShape, srcTy, ofrOffsets, ofrSizes, ofrStrides);
        auto resTy = inferred.cast<MemRefType>();
        return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets,
                                                  ofrSizes, ofrStrides);
      };

      // Collapse RHS head and head_dim (H and D): rhs [B, Lk, H, D] -> [B, Lk, H*D]
      SmallVector<ReassociationIndices, 3> rhsReassoc;
      rhsReassoc.push_back(ReassociationIndices{0});
      rhsReassoc.push_back(ReassociationIndices{1});
      rhsReassoc.push_back(ReassociationIndices{2, 3});
      Value rhsCollapsed = rewriter.create<memref::CollapseShapeOp>(
          makeTaggedLoc("pynq.qmatmul.attn_SV.rhs_collapse"), rhs, rhsReassoc);

      // Collapse output head and head_dim (H and D): out [B, H, D, Lq] -> [B, H*D, Lq]
      SmallVector<ReassociationIndices, 3> outReassoc;
      outReassoc.push_back(ReassociationIndices{0});
      outReassoc.push_back(ReassociationIndices{1, 2});
      outReassoc.push_back(ReassociationIndices{3});
      Value outCollapsed = rewriter.create<memref::CollapseShapeOp>(
          makeTaggedLoc("pynq.qmatmul.attn_SV.out_collapse"), output, outReassoc);

      // Tile counts:
      // - input_tiles: based on Lq (last dim of lhs)
      // - weight_tiles: based on single-head head_dim D
      int64_t inputTileCountI64 = ceilDiv(M, static_cast<int64_t>(pynq::InstrConfig::kTileSize));
      int64_t weightTileCountI64 = ceilDiv(D, static_cast<int64_t>(pynq::InstrConfig::kTileSize));
      if (inputTileCountI64 < 1 || inputTileCountI64 > pynq::InstrConfig::kMaxTileCount)
        return rewriter.notifyMatchFailure(op, "input_tile_count out of hardware range [1, 8]");
      if (weightTileCountI64 < 1 || weightTileCountI64 > pynq::InstrConfig::kMaxTileCount)
        return rewriter.notifyMatchFailure(op, "weight_tile_count out of hardware range [1, 8]");

      Value inputTileCountVal = createI32Const(static_cast<int32_t>(inputTileCountI64));
      Value weightTileCountVal = createI32Const(static_cast<int32_t>(weightTileCountI64));
      Value reduceKVal = createI32Const(static_cast<int32_t>(reduceK));
      Value headTileAxisVal = createI32Const(1); // per attn.SV rule
      Value enableBiasVal = createI32Const(0);
      Value enableTransposeVal = createI32Const(outIsTransposed ? 1 : 0);

      // Copy RHS buffers: per (b, head_group)
      for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t hg = 0; hg < numHeadGroups; ++hg) {
          int64_t hStart = hg * headsPerGroup;
          int64_t hCount = std::min<int64_t>(headsPerGroup, H - hStart);
          if (hCount < 1)
            continue;
          int64_t kOffset = hStart * D;
          int64_t kSize = hCount * D;

          // rhsCollapsed: [B, K, H*D] -> take full K and packed heads columns
          SmallVector<int64_t, 4> offsets = {b, 0, kOffset};
          SmallVector<int64_t, 4> sizes = {1, K, kSize};
          Location copyWLoc = makeTaggedLoc("pynq.qmatmul.attn_SV.copy_rhs");
          Value rhsSubview = makeRankReducedSubview(
              rhsCollapsed, /*resultShape=*/{K, kSize}, offsets, sizes, copyWLoc);

          int64_t rhsLinear = b * numHeadGroups + hg;
          rewriter.create<pynq_ops::CopyOp>(copyWLoc, rhsSubview, rhsBufs[rhsLinear]);
        }
      }

      // Copy LHS buffers: per (b, h, mTile)
      for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
            int64_t mStart = mTile * tileM;
            int64_t mLen = std::min<int64_t>(tileM, M - mStart);
            if (mLen < 1)
              continue;
            SmallVector<int64_t, 4> offsets = {b, h, 0, mStart};
            SmallVector<int64_t, 4> sizes = {1, 1, K, mLen};
            Location copyInLoc = makeTaggedLoc("pynq.qmatmul.attn_SV.copy_lhs");
            Value lhsSubview = makeRankReducedSubview(
                lhs, /*resultShape=*/{K, mLen}, offsets, sizes, copyInLoc);

            int64_t lhsLinear = (b * H + h) * numTilesM + mTile;
            rewriter.create<pynq_ops::CopyOp>(copyInLoc, lhsSubview, lhsBufs[lhsLinear]);
          }
        }
      }

      // Compute: for each head-group/mTile, emit matmul per head (head_idx increments)
      for (int64_t b = 0; b < batchCount; ++b) {
        for (int64_t hg = 0; hg < numHeadGroups; ++hg) {
          int64_t hStart = hg * headsPerGroup;
          int64_t hCount = std::min<int64_t>(headsPerGroup, H - hStart);
          if (hCount < 1)
            continue;
          int64_t kOffset = hStart * D;
          int64_t kSize = hCount * D;

          int64_t rhsLinear = b * numHeadGroups + hg;
          for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
            int64_t mStart = mTile * tileM;
            int64_t mLen = std::min<int64_t>(tileM, M - mStart);
            if (mLen < 1)
              continue;

            int64_t outLinear = (b * numHeadGroups + hg) * numTilesM + mTile;
            for (int64_t localH = 0; localH < hCount; ++localH) {
              int64_t globalH = hStart + localH;
              int64_t lhsLinear = (b * H + globalH) * numTilesM + mTile;
              Value headIdxVal = createI32Const(static_cast<int32_t>(localH));

              std::string tag = ("pynq.qmatmul.attn_SV.compute.b" + std::to_string(b) +
                                 ".hg" + std::to_string(hg) +
                                 ".h" + std::to_string(globalH) +
                                 ".m" + std::to_string(mTile));
              Location computeLoc = makeTaggedLoc(tag);

              rewriter.create<pynq_ops::MatMulOp>(
                  computeLoc, fusedScale,
                  lhsBufs[lhsLinear], rhsBufs[rhsLinear], outBufs[outLinear],
                  inputTileCountVal, weightTileCountVal,
                  reduceKVal, headIdxVal, headTileAxisVal,
                  enableBiasVal, enableTransposeVal);
              rewriter.create<pynq_ops::SyncOp>(makeTaggedLoc("pynq.qmatmul.attn_SV.sync"));
            }

            // Copy packed output (heads in this group) back in one shot.
            // outCollapsed: [B, H*D, Lq]
            SmallVector<int64_t, 3> outOffsets = {b, kOffset, mStart};
            SmallVector<int64_t, 3> outSizes = {1, kSize, mLen};
            Location copyOutLoc = makeTaggedLoc("pynq.qmatmul.attn_SV.copy_out");
            Value outSubview = makeRankReducedSubview(
                outCollapsed, /*resultShape=*/{kSize, mLen}, outOffsets, outSizes, copyOutLoc);
            rewriter.create<pynq_ops::CopyOp>(copyOutLoc, outBufs[outLinear], outSubview);
          }
        }
      }

      rewriter.eraseOp(op);
      return success();
    }

    else {
      return rewriter.notifyMatchFailure(op, "unsupported vivado.qlinear consumer op for PYNQ lowering");
    }
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qlinear -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQLinearToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QLinearOp> {
  using OpRewritePattern<vivado_ops::QLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QLinearOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Tag all lowered ops with a fused location so we can recognize the
    // vivado.qlinear expansion boundaries without introducing marker ops.
    Location loweredLoc = FusedLoc::get(
      rewriter.getContext(),
      {loc, NameLoc::get(rewriter.getStringAttr("pynq.lowered_from_vivado.qlinear"))});
    
    // Extract operands
    Value output = op.getOutput();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    Value fusedScale = op.getFscl();
    
    // Get tile configuration
    int32_t tileM = op.getTileM();
    int32_t tileN = op.getTileN();
    int32_t tileK = op.getTileK();
    
    //===------------------------------------------------------------------===//
    // Step 1: Verify shapes can fit in buffers
    //===------------------------------------------------------------------===//

    bool transposeMode = op.getTransposeMode();
    bool outIsTransposed = op.getIsTransposed();
    bool fuse_bias = op.getFuseBias();

    // Lowering to PYNQ backend only supports transpose_mode=true.
    if (!transposeMode) {
      op.emitOpError("LowerVivadoToPYNQ: vivado.qlinear requires transpose_mode=true for PYNQ backend");
      return failure();
    }
    
    auto inputType = input.getType().dyn_cast<ShapedType>();
    auto weightType = weight.getType().dyn_cast<ShapedType>();
    auto outputType = output.getType().dyn_cast<ShapedType>();
    
    if (!inputType || !weightType || !outputType)
      return rewriter.notifyMatchFailure(op, "operands must have shaped types");

    // We rely on memref.subview and pynq.copy endpoints, so enforce memrefs.
    if (!input.getType().isa<MemRefType>() || !weight.getType().isa<MemRefType>() ||
        !output.getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(op, "expected input/weight/output to be memrefs");
    }
    
    if (!inputType.hasStaticShape() || !weightType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not yet supported");
    
    // Note: QLinear input/output may include batch dimensions. We assume the
    // last two dimensions follow the transpose-mode conventions documented
    // above.
    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();
    auto outputShape = outputType.getShape();
    
    if (inputShape.size() < 2 || outputShape.size() < 2) {
      return rewriter.notifyMatchFailure(op, "expected input/output rank >= 2");
    }
    if (weightShape.size() != 2) {
      return rewriter.notifyMatchFailure(op, "expected weight rank == 2");
    }
    if (outputShape.size() != inputShape.size()) {
      return rewriter.notifyMatchFailure(op, "expected output rank to match input rank");
    }
    // Validate batch dimensions match for input/output.
    for (size_t dim = 0; dim + 2 < inputShape.size(); ++dim) {
      if (inputShape[dim] != outputShape[dim]) {
        return rewriter.notifyMatchFailure(op, "batch dimensions mismatch between input and output");
      }
    }

    // Transpose-mode only:
    //   input  [..., K, M]
    //   weight [K, N]
    //   output [..., M, N] when is_transposed=false
    //   output [..., N, M] when is_transposed=true
    int64_t K = inputShape[inputShape.size() - 2];
    int64_t M = inputShape[inputShape.size() - 1];
    int64_t K_weight = weightShape[0];
    int64_t N = weightShape[1];

    if (K != K_weight) {
      return rewriter.notifyMatchFailure(op, "input/weight dimension mismatch");
    }

    if (!outIsTransposed) {
      if (outputShape[outputShape.size() - 2] != M ||
          outputShape[outputShape.size() - 1] != N) {
        return rewriter.notifyMatchFailure(op, "output shape mismatch");
      }
    } else {
      if (outputShape[outputShape.size() - 2] != N ||
          outputShape[outputShape.size() - 1] != M) {
        return rewriter.notifyMatchFailure(op, "output shape mismatch");
      }
    }
    
    // K must fit in the hardware reduction dimension configured by tileK.
    // If K is larger, it indicates earlier tiling/padding did not run as expected.
    if (K > tileK) {
      op.emitOpError("LowerVivadoToPYNQ: K dimension (") << K
                     << ") exceeds tile_k (" << tileK
                     << "); expected K <= tile_k after earlier passes";
      return failure();
    }
    
    // Calculate buffer capacity needed (in bytes)
    // Each tile is tileSize x tileSize x sizeof(i8)
    auto elementType = inputType.getElementType();
    if (!elementType.isInteger(8)) {
      return rewriter.notifyMatchFailure(op, "only i8 element type supported");
    }

    // PYNQ matmul instruction carries a scale memref for mid-level passes.
    // It is not encoded into the final instruction, but must be a memref in IR.
    if (!fusedScale || !fusedScale.getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(op, "expected fscl (fused_scale) to be a memref for PYNQ binding");
    }

    // Broadcast scalar/len-1 fused_scale along token dimension.
    // Under transpose_mode=true:
    //   output [..., M, N] when is_transposed=false  -> token dim is rank-2
    //   output [..., N, M] when is_transposed=true   -> token dim is rank-1
    int64_t tokenLen = outIsTransposed
                           ? outputShape[outputShape.size() - 1]
                           : outputShape[outputShape.size() - 2];
    if (auto bcast = broadcastLen1ScaleTo(fusedScale, tokenLen, "qlinear_fscl", op, rewriter);
        succeeded(bcast)) {
      fusedScale = *bcast;
    } else {
      return failure();
    }
    
    int64_t inputTileBytes = tileM * tileK * 1;  // i8 = 1 byte
    int64_t weightTileBytes = tileN * tileK * 1;
    int64_t outputTileBytes = tileM * tileN * 1;

    auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t {
      if (b == 0)
        return 0;
      return (a + b - 1) / b;
    };

    // Tile counts (excluding batch dims). M/N may be larger than tileM/tileN.
    int64_t numTilesM = ceilDiv(M, tileM);
    int64_t numTilesN = ceilDiv(N, tileN);
    int64_t numTilesOut = numTilesM * numTilesN;
    
    // Validate tile sizes fit in buffer capacity
    if (inputTileBytes > pynq::BufferConfig::kDefaultCapacityBytes || 
        weightTileBytes > pynq::BufferConfig::kDefaultCapacityBytes ||
        outputTileBytes > pynq::BufferConfig::kDefaultCapacityBytes) {
      return rewriter.notifyMatchFailure(op, 
        "tile size exceeds buffer capacity (adjust tiling)");
    }
    
    //===------------------------------------------------------------------===//
    // Step 2: Allocate virtual buffers for input, weight, output
    //===------------------------------------------------------------------===//
    
    // Create virtual buffers (IDs will be assigned by allocation pass later)
    auto i8Type = rewriter.getIntegerType(8);

    // PYNQ on-chip buffers have a fixed max capacity (64KB). Large tensors are
    // handled by creating multiple buffers and issuing multiple instructions.
    // Per your current assumption, we do NOT limit buffer count here (treat as
    // unlimited resources). Scheduling / reuse will be handled later.
    int64_t batchRank = static_cast<int64_t>(inputShape.size()) - 2;
    int64_t batchCount = 1;
    for (int64_t d = 0; d < batchRank; ++d)
      batchCount *= inputShape[d];

    int64_t inputBufCount = std::max<int64_t>(int64_t{1}, batchCount * numTilesM);
    int64_t weightBufCount = std::max<int64_t>(int64_t{1}, numTilesN);
    int64_t outputBufCount = std::max<int64_t>(int64_t{1}, batchCount * numTilesOut);

    auto inputBufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);
    auto weightBufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);
    auto outputBufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

    SmallVector<Value, 4> inputBufs;
    SmallVector<Value, 4> weightBufs;
    SmallVector<Value, 4> outputBufs;
    inputBufs.reserve(inputBufCount);
    weightBufs.reserve(weightBufCount);
    outputBufs.reserve(outputBufCount);

    for (int64_t i = 0; i < inputBufCount; ++i) {
      inputBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, inputBufferType, rewriter.getStringAttr("input")));
    }
    for (int64_t i = 0; i < weightBufCount; ++i) {
      weightBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, weightBufferType, rewriter.getStringAttr("weight")));
    }
    for (int64_t i = 0; i < outputBufCount; ++i) {
      outputBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, outputBufferType, rewriter.getStringAttr("output")));
    }

    // Optional bias buffers (one per N-tile). Bias element type may differ from i8.
    SmallVector<Value, 4> biasBufs;
    if (bias && fuse_bias) {
      auto biasType = bias.getType().dyn_cast<ShapedType>();
      if (!biasType || !biasType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(op, "bias must have a static shaped type");
      }
      if (!bias.getType().isa<MemRefType>()) {
        return rewriter.notifyMatchFailure(op, "expected bias to be a memref");
      }
      auto biasElemTy = biasType.getElementType();
      if (!biasElemTy.isa<IntegerType, FloatType>()) {
        return rewriter.notifyMatchFailure(op, "unsupported bias element type");
      }
      unsigned biasElemBits = biasElemTy.getIntOrFloatBitWidth();
      if (biasElemBits % 8 != 0) {
        return rewriter.notifyMatchFailure(op, "bias element size must be byte-aligned");
      }
      int64_t biasElemBytes = static_cast<int64_t>(biasElemBits / 8);
      int64_t biasTileBytes = tileN * biasElemBytes;
      if (biasTileBytes > pynq::BufferConfig::kDefaultCapacityBytes) {
        return rewriter.notifyMatchFailure(op, "bias tile size exceeds buffer capacity");
      }
      auto biasBufferType = pynq_ops::BufferType::getVirtual(
          rewriter.getContext(), biasElemTy, pynq::BufferConfig::kDefaultCapacityBytes);
        int64_t biasBufCount = std::max<int64_t>(int64_t{1}, numTilesN);
        biasBufs.reserve(biasBufCount);
        for (int64_t i = 0; i < biasBufCount; ++i) {
        biasBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, biasBufferType, rewriter.getStringAttr("bias")));
        }
    }
    
    //===------------------------------------------------------------------===//
    // Step 3: Fully unrolled tile copies + instructions (no scf.for)
    //===------------------------------------------------------------------===//

    auto createI32ConstAt = [&](int32_t value, Location l) {
      return rewriter.create<arith::ConstantOp>(
          l, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
    };

    auto createI32Const = [&](int32_t value) {
      return createI32ConstAt(value, loweredLoc);
    };

    auto makeTaggedLoc = [&](StringRef tag) -> Location {
      return FusedLoc::get(
          rewriter.getContext(),
          {loweredLoc, NameLoc::get(rewriter.getStringAttr(tag))});
    };

    auto makeStaticOfr = [&](int64_t v) -> OpFoldResult {
      return rewriter.getIndexAttr(v);
    };

    auto makeRankReducedSubview = [&](Value src, ArrayRef<int64_t> resultShape,
                                      ArrayRef<int64_t> offsets,
                                      ArrayRef<int64_t> sizes,
                                      Location l) -> Value {
      auto srcTy = src.getType().cast<MemRefType>();
      int64_t rank = srcTy.getRank();
      SmallVector<OpFoldResult, 4> ofrOffsets;
      SmallVector<OpFoldResult, 4> ofrSizes;
      SmallVector<OpFoldResult, 4> ofrStrides;
      ofrOffsets.reserve(rank);
      ofrSizes.reserve(rank);
      ofrStrides.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        ofrOffsets.push_back(makeStaticOfr(offsets[i]));
        ofrSizes.push_back(makeStaticOfr(sizes[i]));
        ofrStrides.push_back(makeStaticOfr(1));
      }
      Type inferred = memref::SubViewOp::inferRankReducedResultType(
          resultShape, srcTy, ofrOffsets, ofrSizes, ofrStrides);
      auto resTy = inferred.cast<MemRefType>();
      return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets,
                                                ofrSizes, ofrStrides);
    };

    auto makeSubview = [&](Value src, ArrayRef<int64_t> offsets,
                           ArrayRef<int64_t> sizes,
                           Location l) -> Value {
      auto srcTy = src.getType().cast<MemRefType>();
      int64_t rank = srcTy.getRank();
      SmallVector<OpFoldResult, 4> ofrOffsets;
      SmallVector<OpFoldResult, 4> ofrSizes;
      SmallVector<OpFoldResult, 4> ofrStrides;
      ofrOffsets.reserve(rank);
      ofrSizes.reserve(rank);
      ofrStrides.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        ofrOffsets.push_back(makeStaticOfr(offsets[i]));
        ofrSizes.push_back(makeStaticOfr(sizes[i]));
        ofrStrides.push_back(makeStaticOfr(1));
      }
      Type inferred = memref::SubViewOp::inferResultType(srcTy, ofrOffsets,
                                                         ofrSizes, ofrStrides);
      auto resTy = inferred.cast<MemRefType>();
      return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets,
                                                ofrSizes, ofrStrides);
    };

    // Helper to decode a linear batch index into per-dim indices.
    auto decodeBatchIndex = [&](int64_t linear) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> idx;
      idx.resize(batchRank, 0);
      for (int64_t d = batchRank; d-- > 0;) {
        int64_t dim = inputShape[d];
        idx[d] = linear % dim;
        linear /= dim;
      }
      return idx;
    };

    // 3.1 Copy weight tiles (and bias tiles) into per-tile buffers.
    for (int64_t nTile = 0; nTile < numTilesN; ++nTile) {
      int64_t nStart = nTile * tileN;
      int64_t nLen = std::min<int64_t>(tileN, N - nStart);
      Location wLoc = makeTaggedLoc("pynq.qlinear.copy_weight");
      Value wSubview = makeSubview(weight, /*offsets=*/{0, nStart},
                                   /*sizes=*/{K, nLen}, wLoc);
      rewriter.create<pynq_ops::CopyOp>(wLoc, wSubview, weightBufs[nTile]);

      if (bias) {
        auto biasTy = bias.getType().cast<MemRefType>();
        Location bLoc = makeTaggedLoc("pynq.qlinear.copy_bias");
        if (biasTy.getRank() == 1) {
          Value bSubview = makeSubview(bias, /*offsets=*/{nStart},
                                       /*sizes=*/{nLen}, bLoc);
          rewriter.create<pynq_ops::CopyOp>(bLoc, bSubview, biasBufs[nTile]);
        } else if (biasTy.getRank() == 2) {
          // Treat rank-2 bias as [1, N]-like and rank-reduce to 1D.
          auto biasShape = biasTy.getShape();
          if (biasShape[0] != 1) {
            return rewriter.notifyMatchFailure(op, "expected rank-2 bias to have leading dim == 1");
          }
          Value bSubview = makeRankReducedSubview(
              bias, /*resultShape=*/{nLen},
              /*offsets=*/{0, nStart}, /*sizes=*/{1, nLen}, bLoc);
          rewriter.create<pynq_ops::CopyOp>(bLoc, bSubview, biasBufs[nTile]);
        } else {
          return rewriter.notifyMatchFailure(op, "unsupported bias rank for PYNQ lowering");
        }
      }
    }

    // 3.2 Copy input tiles (per batch x M-tile) into buffers.
    for (int64_t b = 0; b < batchCount; ++b) {
      SmallVector<int64_t, 4> batchIdx = decodeBatchIndex(b);
      for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
        int64_t mStart = mTile * tileM;
        int64_t mLen = std::min<int64_t>(tileM, M - mStart);

        SmallVector<int64_t, 8> inOffsets;
        SmallVector<int64_t, 8> inSizes;
        inOffsets.reserve(inputShape.size());
        inSizes.reserve(inputShape.size());
        for (int64_t d = 0; d < batchRank; ++d) {
          inOffsets.push_back(batchIdx[d]);
          inSizes.push_back(1);
        }
        // transpose_mode=true input tail dims are [K, M].
        inOffsets.push_back(0);
        inOffsets.push_back(mStart);
        inSizes.push_back(K);
        inSizes.push_back(mLen);

        Location inLoc = makeTaggedLoc("pynq.qlinear.copy_input");
        Value inSubview = makeRankReducedSubview(input, /*resultShape=*/{K, mLen},
                                                 inOffsets, inSizes, inLoc);
        int64_t inputBufLinear = b * numTilesM + mTile;
        rewriter.create<pynq_ops::CopyOp>(inLoc, inSubview, inputBufs[inputBufLinear]);
      }
    }

    // 3.3 Issue matmul ops per output tile and copy back to output.
    // NOTE: We use abstract pynq.buffer operands here. A later pass is
    // expected to assign/pack them to the physical [0..7] range.

    // NOTE: tileM/tileN are the 2D tiling factors for slicing tensors.
    // input_tile_count/weight_tile_count are *instruction encoding* fields and
    // count how many InstrConfig::kTileSize chunks are present along the last
    // dimension of the per-tile buffers. Padding is expected to have run.
    // IMPORTANT: These values must reflect the actual per-tile extents (mLen/nLen)
    // rather than the configured tileM/tileN, otherwise boundary tiles may be
    // over-counted and lead to incorrect behavior.
    Value reduceKVal = createI32Const(static_cast<int32_t>(K));
    Value headIdxVal = createI32Const(0); // default 0
    Value headTileAxisVal = createI32Const(0);
    Value enableBiasVal = createI32Const(0);
    // In transpose_mode=true lowering, enable_transpose reflects whether the
    // output layout is transposed (is_transposed).
    Value enableTransposeVal = createI32Const(outIsTransposed ? 1 : 0);

    for (int64_t b = 0; b < batchCount; ++b) {
      SmallVector<int64_t, 4> batchIdx = decodeBatchIndex(b);
      for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
        int64_t mStart = mTile * tileM;
        int64_t mLen = std::min<int64_t>(tileM, M - mStart);
        int64_t inputBufLinear = b * numTilesM + mTile;

        int64_t inputTileCountI64 = ceilDiv(
            static_cast<int64_t>(mLen),
            static_cast<int64_t>(pynq::InstrConfig::kTileSize));
        if (inputTileCountI64 < 1 || inputTileCountI64 > pynq::InstrConfig::kMaxTileCount)
          return rewriter.notifyMatchFailure(op, "input_tile_count out of hardware range [1, 8]");

        for (int64_t nTile = 0; nTile < numTilesN; ++nTile) {
          int64_t nStart = nTile * tileN;
          int64_t nLen = std::min<int64_t>(tileN, N - nStart);
          int64_t outTileLinear = b * numTilesOut + mTile * numTilesN + nTile;

          int64_t weightTileCountI64 = ceilDiv(
              static_cast<int64_t>(nLen),
              static_cast<int64_t>(pynq::InstrConfig::kTileSize));
          if (weightTileCountI64 < 1 || weightTileCountI64 > pynq::InstrConfig::kMaxTileCount)
            return rewriter.notifyMatchFailure(op, "weight_tile_count out of hardware range [1, 8]");

          // Compute marker via loc tags (no dedicated marker ops).
          std::string tag = ("pynq.qlinear.compute.b" + std::to_string(b) +
                             ".m" + std::to_string(mTile) +
                             ".n" + std::to_string(nTile));
          Location computeLoc = makeTaggedLoc(tag);

          Value inputTileCountVal = createI32ConstAt(static_cast<int32_t>(inputTileCountI64), computeLoc);
          Value weightTileCountVal = createI32ConstAt(static_cast<int32_t>(weightTileCountI64), computeLoc);

            rewriter.create<pynq_ops::MatMulOp>(
              computeLoc, fusedScale,
              inputBufs[inputBufLinear], weightBufs[nTile], outputBufs[outTileLinear],
              inputTileCountVal, weightTileCountVal,
              reduceKVal, headIdxVal, headTileAxisVal,
              enableBiasVal, enableTransposeVal);

          rewriter.create<pynq_ops::SyncOp>(makeTaggedLoc("pynq.qlinear.sync"));

          // Copy output tile back to host memref via a rank-reduced subview.
          SmallVector<int64_t, 8> outOffsets;
          SmallVector<int64_t, 8> outSizes;
          outOffsets.reserve(outputShape.size());
          outSizes.reserve(outputShape.size());
          for (int64_t d = 0; d < batchRank; ++d) {
            outOffsets.push_back(batchIdx[d]);
            outSizes.push_back(1);
          }

          if (!outIsTransposed) {
            outOffsets.push_back(mStart);
            outOffsets.push_back(nStart);
            outSizes.push_back(mLen);
            outSizes.push_back(nLen);
            Value outSubview = makeRankReducedSubview(
                output, /*resultShape=*/{mLen, nLen}, outOffsets, outSizes,
                makeTaggedLoc("pynq.qlinear.copy_output"));
            rewriter.create<pynq_ops::CopyOp>(makeTaggedLoc("pynq.qlinear.copy_output"),
                                             outputBufs[outTileLinear], outSubview);
          } else {
            outOffsets.push_back(nStart);
            outOffsets.push_back(mStart);
            outSizes.push_back(nLen);
            outSizes.push_back(mLen);
            Value outSubview = makeRankReducedSubview(
                output, /*resultShape=*/{nLen, mLen}, outOffsets, outSizes,
                makeTaggedLoc("pynq.qlinear.copy_output"));
            rewriter.create<pynq_ops::CopyOp>(makeTaggedLoc("pynq.qlinear.copy_output"),
                                             outputBufs[outTileLinear], outSubview);
          }
        }
      }
    }
    
    //===------------------------------------------------------------------===//
    // Step 11: Remove original vivado.qlinear op
    //===------------------------------------------------------------------===//
    
    rewriter.eraseOp(op);
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qadd -> pynq.qadd
//===----------------------------------------------------------------------===//

struct VivadoQAddToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QAddOp> {
  using OpRewritePattern<vivado_ops::QAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QAddOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Location loweredLoc = FusedLoc::get(
        rewriter.getContext(),
        {loc, NameLoc::get(rewriter.getStringAttr(
                  "pynq.lowered_from_vivado.qadd"))});

    Value output = op.getOutput();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value xScale = op.getXScale();
    Value yScale = op.getYScale();
    Value oScaleInv = op.getOScaleInv();
    Value oScale = op.getOScale();

    // PYNQ qadd lowering requirements:
    // - transpose_mode must be enabled (matches the rest of PYNQ vector ops)
    // NOTE: qadd is element-wise and does not depend on whether the underlying
    //       memref is in transposed layout; do not require is_transposed=true.
    bool transposeMode = op.getTransposeMode();
    bool outIsTransposed = op.getIsTransposed();
    if (!transposeMode)
      return rewriter.notifyMatchFailure(op, "expected transpose_mode=true for PYNQ qadd");

    // Zero points are not supported by pynq.qadd (only scale bindings).
    if (op.getXZero() || op.getYZero() || op.getOZero())
      return rewriter.notifyMatchFailure(op, "zero points not supported for PYNQ qadd");

    auto lhsType = lhs.getType().dyn_cast<ShapedType>();
    auto rhsType = rhs.getType().dyn_cast<ShapedType>();
    auto outType = output.getType().dyn_cast<ShapedType>();
    if (!lhsType || !rhsType || !outType)
      return rewriter.notifyMatchFailure(op, "expected shaped lhs/rhs/output");

    // We rely on memref.subview and pynq.copy endpoints, so enforce memrefs.
    if (!lhs.getType().isa<MemRefType>() || !rhs.getType().isa<MemRefType>() ||
        !output.getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(op, "expected lhs/rhs/output to be memrefs");
    }

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes for PYNQ lowering");

    if (lhsType.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "expected rank >= 2 for qadd tiling");
    if (lhsType.getRank() != rhsType.getRank() || lhsType.getRank() != outType.getRank())
      return rewriter.notifyMatchFailure(op, "lhs/rhs/output rank mismatch");
    if (lhsType.getShape() != rhsType.getShape() || lhsType.getShape() != outType.getShape())
      return rewriter.notifyMatchFailure(op, "expected lhs/rhs/output to have identical shapes");

    auto elemTy = lhsType.getElementType();
    if (!elemTy.isInteger(8))
      return rewriter.notifyMatchFailure(op, "expected i8 element type for qadd lowering");
    if (rhsType.getElementType() != elemTy || outType.getElementType() != elemTy)
      return rewriter.notifyMatchFailure(op, "expected rhs/output element type to match lhs");

    // Scale bindings: must be memrefs (host-side bindings).
    if (!xScale || !xScale.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected x_scale to be a memref for pynq.qadd");
    if (!yScale || !yScale.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected y_scale to be a memref for pynq.qadd");
    if (!oScaleInv || !oScaleInv.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected o_scale_inv to be a memref for pynq.qadd");
    // o_scale exists on vivado.qadd but is not consumed by pynq.qadd; still
    // validate it is a memref so we don't silently drop a non-memref value.
    if (!oScale || !oScale.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected o_scale to be a memref for qadd (unused in pynq)");

    auto shape = lhsType.getShape();
    int64_t rank = lhsType.getRank();
    int64_t batchRank = rank - 2;
    int64_t M = shape[rank - 2];
    int64_t N = shape[rank - 1];

    // Broadcast scalar/len-1 scales along token dimension.
    // When is_transposed=false, token dim is rank-2; when is_transposed=true,
    // token dim is rank-1.
    int64_t tokenLen = outIsTransposed ? N : M;
    if (auto bcast = broadcastLen1ScaleTo(xScale, tokenLen, "qadd_x_scale", op, rewriter);
        succeeded(bcast)) {
      xScale = *bcast;
    } else {
      return failure();
    }
    if (auto bcast = broadcastLen1ScaleTo(yScale, tokenLen, "qadd_y_scale", op, rewriter);
        succeeded(bcast)) {
      yScale = *bcast;
    } else {
      return failure();
    }
    if (auto bcast = broadcastLen1ScaleTo(oScaleInv, tokenLen, "qadd_o_scale_inv", op, rewriter);
        succeeded(bcast)) {
      oScaleInv = *bcast;
    } else {
      return failure();
    }
    // o_scale is unused in pynq.qadd but keep it consistent for downstream.
    if (auto bcast = broadcastLen1ScaleTo(oScale, tokenLen, "qadd_o_scale", op, rewriter);
        succeeded(bcast)) {
      oScale = *bcast;
    } else {
      return failure();
    }

    // Use default hardware tile sizes for 2D slicing.
    int32_t tileM = pynq::TileConfig::kDefaultTileM;
    int32_t tileN = pynq::TileConfig::kDefaultTileN;

    if (M <= 0 || N <= 0)
      return rewriter.notifyMatchFailure(op, "expected non-empty M/N dimensions");
    if (tileM < 1 || tileN < 1)
      return rewriter.notifyMatchFailure(op, "invalid tileM/tileN configuration");

    if (tileM > static_cast<int32_t>(pynq::InstrConfig::kMaxReduceK))
      return rewriter.notifyMatchFailure(op, "tileM exceeds hardware reduce_k limit");

    int64_t tileBytes = static_cast<int64_t>(tileM) * tileN * 1;
    if (tileBytes > pynq::BufferConfig::kDefaultCapacityBytes)
      return rewriter.notifyMatchFailure(op, "tile does not fit in PYNQ buffer capacity");

    auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };
    int64_t numTilesM = ceilDiv(M, tileM);
    int64_t numTilesN = ceilDiv(N, tileN);

    int64_t batchCount = 1;
    for (int64_t d = 0; d < batchRank; ++d)
      batchCount *= shape[d];

    int64_t bufCount = std::max<int64_t>(int64_t{1}, batchCount * numTilesM * numTilesN);

    auto i8Type = rewriter.getIntegerType(8);
    auto bufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

    // Allocate one primary buffer (lhs/output, in-place) and one extra buffer
    // (rhs, read-only) per tile.
    SmallVector<Value, 4> lhsBufs;
    SmallVector<Value, 4> rhsBufs;
    lhsBufs.reserve(bufCount);
    rhsBufs.reserve(bufCount);
    for (int64_t i = 0; i < bufCount; ++i) {
      lhsBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, bufferType, rewriter.getStringAttr("activation")));
      rhsBufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, bufferType, rewriter.getStringAttr("activation")));
    }

    auto createI32ConstAt = [&](int32_t value, Location l) {
      return rewriter.create<arith::ConstantOp>(
          l, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
    };
    auto createI32Const = [&](int32_t value) { return createI32ConstAt(value, loweredLoc); };

    auto makeTaggedLoc = [&](StringRef tag) -> Location {
      return FusedLoc::get(
          rewriter.getContext(),
          {loweredLoc, NameLoc::get(rewriter.getStringAttr(tag))});
    };

    auto makeStaticOfr = [&](int64_t v) -> OpFoldResult {
      return rewriter.getIndexAttr(v);
    };

    auto makeRankReducedSubview = [&](Value src, ArrayRef<int64_t> resultShape,
                                      ArrayRef<int64_t> offsets,
                                      ArrayRef<int64_t> sizes,
                                      Location l) -> Value {
      auto srcTy = src.getType().cast<MemRefType>();
      int64_t r = srcTy.getRank();
      SmallVector<OpFoldResult, 4> ofrOffsets;
      SmallVector<OpFoldResult, 4> ofrSizes;
      SmallVector<OpFoldResult, 4> ofrStrides;
      ofrOffsets.reserve(r);
      ofrSizes.reserve(r);
      ofrStrides.reserve(r);
      for (int64_t i = 0; i < r; ++i) {
        ofrOffsets.push_back(makeStaticOfr(offsets[i]));
        ofrSizes.push_back(makeStaticOfr(sizes[i]));
        ofrStrides.push_back(makeStaticOfr(1));
      }
      Type inferred = memref::SubViewOp::inferRankReducedResultType(
          resultShape, srcTy, ofrOffsets, ofrSizes, ofrStrides);
      auto resTy = inferred.cast<MemRefType>();
      return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets, ofrSizes,
                                                ofrStrides);
    };

    auto decodeBatchIndex = [&](int64_t linear) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> idx;
      idx.resize(batchRank, 0);
      for (int64_t d = batchRank; d-- > 0;) {
        int64_t dim = shape[d];
        idx[d] = linear % dim;
        linear /= dim;
      }
      return idx;
    };

    // Fully unrolled: for each tile, move lhs/rhs -> qadd -> move lhs back.
    for (int64_t b = 0; b < batchCount; ++b) {
      SmallVector<int64_t, 4> batchIdx = decodeBatchIndex(b);
      for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
        int64_t mStart = mTile * tileM;
        int64_t mLen = std::min<int64_t>(tileM, M - mStart);
        if (mLen < 1 || mLen > static_cast<int64_t>(pynq::InstrConfig::kMaxReduceK))
          return rewriter.notifyMatchFailure(op, "reduce_k out of hardware range [1, 256]");

        for (int64_t nTile = 0; nTile < numTilesN; ++nTile) {
          int64_t nStart = nTile * tileN;
          int64_t nLen = std::min<int64_t>(tileN, N - nStart);
          if (nLen < 1)
            continue;

          int64_t tileCountI64 = ceilDiv(
              static_cast<int64_t>(nLen),
              static_cast<int64_t>(pynq::InstrConfig::kTileSize));
          if (tileCountI64 < 1 || tileCountI64 > pynq::InstrConfig::kMaxTileCount)
            return rewriter.notifyMatchFailure(op, "tile_count out of hardware range [1, 8]");

          int64_t linear = b * (numTilesM * numTilesN) + mTile * numTilesN + nTile;

          SmallVector<int64_t, 8> offsets;
          SmallVector<int64_t, 8> sizes;
          offsets.reserve(rank);
          sizes.reserve(rank);
          for (int64_t d = 0; d < batchRank; ++d) {
            offsets.push_back(batchIdx[d]);
            sizes.push_back(1);
          }
          offsets.push_back(mStart);
          offsets.push_back(nStart);
          sizes.push_back(mLen);
          sizes.push_back(nLen);

          Location copyLhsLoc = makeTaggedLoc("pynq.qadd.copy_lhs");
          Value lhsSubview = makeRankReducedSubview(
              lhs, /*resultShape=*/{mLen, nLen}, offsets, sizes, copyLhsLoc);
          rewriter.create<pynq_ops::CopyOp>(copyLhsLoc, lhsSubview, lhsBufs[linear]);

          Location copyRhsLoc = makeTaggedLoc("pynq.qadd.copy_rhs");
          Value rhsSubview = makeRankReducedSubview(
              rhs, /*resultShape=*/{mLen, nLen}, offsets, sizes, copyRhsLoc);
          rewriter.create<pynq_ops::CopyOp>(copyRhsLoc, rhsSubview, rhsBufs[linear]);

          std::string tag = ("pynq.qadd.compute.b" + std::to_string(b) +
                             ".m" + std::to_string(mTile) +
                             ".n" + std::to_string(nTile));
          Location computeLoc = makeTaggedLoc(tag);
          Value tileCountVal = createI32Const(static_cast<int32_t>(tileCountI64));
          Value reduceKVal = createI32Const(static_cast<int32_t>(mLen));

          // In-place: primary buffer (lhsBufs[linear]) is both input and output.
          // rhsBufs[linear] is read-only and wired to extra_buffer.
          rewriter.create<pynq_ops::QAddOp>(
              computeLoc,
              xScale.cast<TypedValue<MemRefType>>(),
              yScale.cast<TypedValue<MemRefType>>(),
              oScaleInv.cast<TypedValue<MemRefType>>(),
              lhsBufs[linear],
              tileCountVal,
              reduceKVal,
              rhsBufs[linear]);
          rewriter.create<pynq_ops::SyncOp>(makeTaggedLoc("pynq.qadd.sync"));

          Location copyOutLoc = makeTaggedLoc("pynq.qadd.copy_out");
          Value outSubview = makeRankReducedSubview(
              output, /*resultShape=*/{mLen, nLen}, offsets, sizes, copyOutLoc);
          rewriter.create<pynq_ops::CopyOp>(copyOutLoc, lhsBufs[linear], outSubview);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.int_gelu -> pynq.gelu
//===----------------------------------------------------------------------===//

struct VivadoIntGELUToPYNQPattern 
    : public OpRewritePattern<vivado_ops::IntGELUOp> {
  using OpRewritePattern<vivado_ops::IntGELUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::IntGELUOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Location loweredLoc = FusedLoc::get(
        rewriter.getContext(),
        {loc, NameLoc::get(rewriter.getStringAttr(
                  "pynq.lowered_from_vivado.int_gelu"))});

    Value output = op.getOutput();
    Value input = op.getInput();
    Value iscl = op.getIscl();
    Value osclInv = op.getOsclInv();

    // GELU lowering requirements (transpose-mode only):
    // 1) transpose_mode must be enabled
    // 2) is_transposed must be enabled (backend does not accept output being
    //    transposed relative to input under transpose-mode lowering)
    bool transposeMode = op.getTransposeMode();
    bool isTransposed = op.getIsTransposed();
    if (!transposeMode)
      return rewriter.notifyMatchFailure(op, "expected transpose_mode=true for PYNQ gelu");
    if (!isTransposed)
      return rewriter.notifyMatchFailure(op, "expected is_transposed=true for PYNQ gelu");

    auto inputType = input.getType().dyn_cast<ShapedType>();
    auto outputType = output.getType().dyn_cast<ShapedType>();
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected shaped input/output");

    if (!input.getType().isa<MemRefType>() || !output.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected memref input/output for PYNQ lowering");

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes for PYNQ lowering");

    if (inputType.getRank() < 2 || outputType.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "expected rank >= 2 for GELU tiling");

    if (inputType.getRank() != outputType.getRank())
      return rewriter.notifyMatchFailure(op, "input/output rank mismatch");

    if (inputType.getShape() != outputType.getShape())
      return rewriter.notifyMatchFailure(op, "expected input/output to have identical shapes");

    auto elemTy = inputType.getElementType();
    if (!elemTy.isInteger(8))
      return rewriter.notifyMatchFailure(op, "expected i8 element type for GELU lowering");

    if (outputType.getElementType() != elemTy)
      return rewriter.notifyMatchFailure(op, "expected output element type to match input");

    if (!iscl || !iscl.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected iscl to be a memref for pynq.gelu");
    if (!osclInv || !osclInv.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected oscl_inv to be a memref for pynq.gelu");

    auto shape = inputType.getShape();
    int64_t rank = inputType.getRank();
    int64_t batchRank = rank - 2;

    // Broadcast scalar/len-1 scales along token (last dim) in transpose-mode.
    int64_t tokenLen = shape[rank - 1];
    if (auto bcast = broadcastLen1ScaleTo(iscl, tokenLen, "gelu_iscl", op, rewriter);
        succeeded(bcast)) {
      iscl = *bcast;
    } else {
      return failure();
    }
    if (auto bcast = broadcastLen1ScaleTo(osclInv, tokenLen, "gelu_oscl_inv", op, rewriter);
        succeeded(bcast)) {
      osclInv = *bcast;
    } else {
      return failure();
    }
    int64_t M = shape[rank - 2];
    int64_t N = shape[rank - 1];

    // Use default hardware tile sizes for 2D slicing of the activation.
    // Instruction encoding uses tile_count based on InstrConfig::kTileSize along
    // the last dimension, and reduce_k based on the second-to-last dimension.
    int32_t tileM = pynq::TileConfig::kDefaultTileM;
    int32_t tileN = pynq::TileConfig::kDefaultTileN;

    if (M <= 0 || N <= 0)
      return rewriter.notifyMatchFailure(op, "expected non-empty M/N dimensions");

    if (tileM < 1 || tileN < 1)
      return rewriter.notifyMatchFailure(op, "invalid tileM/tileN configuration");

    if (tileM > static_cast<int32_t>(pynq::InstrConfig::kMaxReduceK))
      return rewriter.notifyMatchFailure(op, "tileM exceeds hardware reduce_k limit");

    int64_t tileBytes = static_cast<int64_t>(tileM) * tileN * 1;
    if (tileBytes > pynq::BufferConfig::kDefaultCapacityBytes)
      return rewriter.notifyMatchFailure(op, "tile does not fit in PYNQ buffer capacity");

    auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t {
      return (a + b - 1) / b;
    };

    int64_t numTilesM = ceilDiv(M, tileM);
    int64_t numTilesN = ceilDiv(N, tileN);

    int64_t batchCount = 1;
    for (int64_t d = 0; d < batchRank; ++d) {
      batchCount *= shape[d];
    }

    int64_t bufCount = std::max<int64_t>(int64_t{1}, batchCount * numTilesM * numTilesN);

    auto i8Type = rewriter.getIntegerType(8);
    auto bufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

    SmallVector<Value, 4> bufs;
    bufs.reserve(bufCount);
    for (int64_t i = 0; i < bufCount; ++i) {
      bufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, bufferType, rewriter.getStringAttr("activation")));
    }

    auto createI32ConstAt = [&](int32_t value, Location l) {
      return rewriter.create<arith::ConstantOp>(
          l, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
    };

    auto createI32Const = [&](int32_t value) { return createI32ConstAt(value, loweredLoc); };

    auto makeTaggedLoc = [&](StringRef tag) -> Location {
      return FusedLoc::get(
          rewriter.getContext(),
          {loweredLoc, NameLoc::get(rewriter.getStringAttr(tag))});
    };

    auto makeStaticOfr = [&](int64_t v) -> OpFoldResult {
      return rewriter.getIndexAttr(v);
    };

    auto makeRankReducedSubview = [&](Value src, ArrayRef<int64_t> resultShape,
                                      ArrayRef<int64_t> offsets,
                                      ArrayRef<int64_t> sizes,
                                      Location l) -> Value {
      auto srcTy = src.getType().cast<MemRefType>();
      int64_t r = srcTy.getRank();
      SmallVector<OpFoldResult, 4> ofrOffsets;
      SmallVector<OpFoldResult, 4> ofrSizes;
      SmallVector<OpFoldResult, 4> ofrStrides;
      ofrOffsets.reserve(r);
      ofrSizes.reserve(r);
      ofrStrides.reserve(r);
      for (int64_t i = 0; i < r; ++i) {
        ofrOffsets.push_back(makeStaticOfr(offsets[i]));
        ofrSizes.push_back(makeStaticOfr(sizes[i]));
        ofrStrides.push_back(makeStaticOfr(1));
      }
      Type inferred = memref::SubViewOp::inferRankReducedResultType(
          resultShape, srcTy, ofrOffsets, ofrSizes, ofrStrides);
      auto resTy = inferred.cast<MemRefType>();
      return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets, ofrSizes,
                                                ofrStrides);
    };

    auto decodeBatchIndex = [&](int64_t linear) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> idx;
      idx.resize(batchRank, 0);
      for (int64_t d = batchRank; d-- > 0;) {
        int64_t dim = shape[d];
        idx[d] = linear % dim;
        linear /= dim;
      }
      return idx;
    };

    for (int64_t b = 0; b < batchCount; ++b) {
      SmallVector<int64_t, 4> batchIdx = decodeBatchIndex(b);
      for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
        int64_t mStart = mTile * tileM;
        int64_t mLen = std::min<int64_t>(tileM, M - mStart);
        if (mLen < 1 || mLen > static_cast<int64_t>(pynq::InstrConfig::kMaxReduceK))
          return rewriter.notifyMatchFailure(op, "reduce_k out of hardware range [1, 256]");

        for (int64_t nTile = 0; nTile < numTilesN; ++nTile) {
          int64_t nStart = nTile * tileN;
          int64_t nLen = std::min<int64_t>(tileN, N - nStart);

          int64_t tileCountI64 = ceilDiv(
              static_cast<int64_t>(nLen),
              static_cast<int64_t>(pynq::InstrConfig::kTileSize));
          if (tileCountI64 < 1 || tileCountI64 > pynq::InstrConfig::kMaxTileCount)
            return rewriter.notifyMatchFailure(op, "tile_count out of hardware range [1, 8]");

          int64_t linear = b * (numTilesM * numTilesN) + mTile * numTilesN + nTile;

          SmallVector<int64_t, 8> offsets;
          SmallVector<int64_t, 8> sizes;
          offsets.reserve(rank);
          sizes.reserve(rank);
          for (int64_t d = 0; d < batchRank; ++d) {
            offsets.push_back(batchIdx[d]);
            sizes.push_back(1);
          }
          offsets.push_back(mStart);
          offsets.push_back(nStart);
          sizes.push_back(mLen);
          sizes.push_back(nLen);

          Location copyInLoc = makeTaggedLoc("pynq.int_gelu.copy_in");
          Value inSubview = makeRankReducedSubview(
              input, /*resultShape=*/{mLen, nLen}, offsets, sizes, copyInLoc);
          rewriter.create<pynq_ops::CopyOp>(copyInLoc, inSubview, bufs[linear]);

            std::string tag = ("pynq.int_gelu.compute.b" + std::to_string(b) +
                     ".m" + std::to_string(mTile) +
                     ".n" + std::to_string(nTile));
            Location computeLoc = makeTaggedLoc(tag);
          Value tileCountVal = createI32Const(static_cast<int32_t>(tileCountI64));
          Value reduceKVal = createI32Const(static_cast<int32_t>(mLen));
          rewriter.create<pynq_ops::GELUOp>(computeLoc,
                                           iscl.cast<TypedValue<MemRefType>>(),
                                           osclInv.cast<TypedValue<MemRefType>>(),
                                           bufs[linear], tileCountVal, reduceKVal);
          rewriter.create<pynq_ops::SyncOp>(makeTaggedLoc("pynq.int_gelu.sync"));

          Location copyOutLoc = makeTaggedLoc("pynq.int_gelu.copy_out");
          Value outSubview = makeRankReducedSubview(
              output, /*resultShape=*/{mLen, nLen}, offsets, sizes, copyOutLoc);
          rewriter.create<pynq_ops::CopyOp>(copyOutLoc, bufs[linear], outSubview);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.int_softmax -> pynq.softmax
//===----------------------------------------------------------------------===//

struct VivadoIntSoftmaxToPYNQPattern 
    : public OpRewritePattern<vivado_ops::IntSoftmaxOp> {
  using OpRewritePattern<vivado_ops::IntSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::IntSoftmaxOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Location loweredLoc = FusedLoc::get(
        rewriter.getContext(),
        {loc, NameLoc::get(rewriter.getStringAttr(
                  "pynq.lowered_from_vivado.int_softmax"))});

    // Softmax lowering requirements (transpose-mode only):
    // 1) transpose_mode must be enabled
    // 2) is_transposed must be enabled (backend does not accept output being
    //    transposed relative to input under transpose-mode lowering)
    bool transposeMode = op.getTransposeMode();
    bool isTransposed = op.getIsTransposed();
    if (!transposeMode)
      return rewriter.notifyMatchFailure(op, "expected transpose_mode=true for PYNQ softmax");
    if (!isTransposed)
      return rewriter.notifyMatchFailure(op, "expected is_transposed=true for PYNQ softmax");

    Value output = op.getOutput();
    Value input = op.getInput();
    Value iscl = op.getIscl();
    Value osclInv = op.getOsclInv();
    int64_t axis = op.getAxis();

    auto inputType = input.getType().dyn_cast<ShapedType>();
    auto outputType = output.getType().dyn_cast<ShapedType>();
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected shaped input/output");

    if (!input.getType().isa<MemRefType>() || !output.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected memref input/output for PYNQ lowering");

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes for PYNQ lowering");

    if (inputType.getRank() < 2 || outputType.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "expected rank >= 2 for softmax tiling");

    if (inputType.getRank() != outputType.getRank())
      return rewriter.notifyMatchFailure(op, "input/output rank mismatch");
    if (inputType.getShape() != outputType.getShape())
      return rewriter.notifyMatchFailure(op, "expected input/output to have identical shapes");

    auto elemTy = inputType.getElementType();
    if (!elemTy.isInteger(8))
      return rewriter.notifyMatchFailure(op, "expected i8 element type for softmax lowering");
    if (outputType.getElementType() != elemTy)
      return rewriter.notifyMatchFailure(op, "expected output element type to match input");

    if (!iscl || !iscl.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected iscl to be a memref for pynq.softmax");
    if (!osclInv || !osclInv.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected oscl_inv to be a memref for pynq.softmax");

    int64_t rank = inputType.getRank();
    // Normalize axis into [0, rank).
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of bounds");

    // In transpose-mode, softmax reduces along the second-to-last dimension.
    if (axis != rank - 2)
      return rewriter.notifyMatchFailure(op, "expected axis == -2 (rank-2) under transpose-mode");

    auto shape = inputType.getShape();
    int64_t batchRank = rank - 2;
    int64_t axisLen = shape[rank - 2];
    int64_t M = shape[rank - 1];

    // Broadcast scalar/len-1 scales along token (last dim) in transpose-mode.
    int64_t tokenLen = M;
    if (auto bcast = broadcastLen1ScaleTo(iscl, tokenLen, "softmax_iscl", op, rewriter);
        succeeded(bcast)) {
      iscl = *bcast;
    } else {
      return failure();
    }
    if (auto bcast = broadcastLen1ScaleTo(osclInv, tokenLen, "softmax_oscl_inv", op, rewriter);
        succeeded(bcast)) {
      osclInv = *bcast;
    } else {
      return failure();
    }

    if (axisLen <= 0 || M <= 0)
      return rewriter.notifyMatchFailure(op, "expected non-empty softmax dimensions");

    // Constraint: cannot tile/reduce-split along axisLen. Require axisLen <= tileN.
    int32_t tileN = pynq::TileConfig::kDefaultTileN;
    if (tileN < 1)
      return rewriter.notifyMatchFailure(op, "invalid tileN configuration");
    if (axisLen > tileN)
      return rewriter.notifyMatchFailure(op, "softmax axis exceeds tileN; reduce axis cannot be tiled");
    if (axisLen > static_cast<int64_t>(pynq::InstrConfig::kMaxReduceK))
      return rewriter.notifyMatchFailure(op, "reduce_k out of hardware range [1, 256]");

    // We may tile along the last dimension (M), similar to GELU.
    int32_t tileM = pynq::TileConfig::kDefaultTileM;
    if (tileM < 1)
      return rewriter.notifyMatchFailure(op, "invalid tileM configuration");

    // Per-tile buffer footprint: axisLen x tileM (i8).
    int64_t tileBytes = axisLen * static_cast<int64_t>(tileM) * 1;
    if (tileBytes > pynq::BufferConfig::kDefaultCapacityBytes)
      return rewriter.notifyMatchFailure(op, "tile does not fit in PYNQ buffer capacity");

    auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };
    int64_t numTilesM = ceilDiv(M, tileM);

    int64_t batchCount = 1;
    for (int64_t d = 0; d < batchRank; ++d)
      batchCount *= shape[d];

    int64_t bufCount = std::max<int64_t>(int64_t{1}, batchCount * numTilesM);

    auto i8Type = rewriter.getIntegerType(8);
    auto bufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

    SmallVector<Value, 4> bufs;
    bufs.reserve(bufCount);
    for (int64_t i = 0; i < bufCount; ++i) {
      bufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, bufferType, rewriter.getStringAttr("activation")));
    }

    auto createI32ConstAt = [&](int32_t value, Location l) {
      return rewriter.create<arith::ConstantOp>(
          l, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
    };
    auto createI32Const = [&](int32_t value) { return createI32ConstAt(value, loweredLoc); };

    auto makeTaggedLoc = [&](StringRef tag) -> Location {
      return FusedLoc::get(
          rewriter.getContext(),
          {loweredLoc, NameLoc::get(rewriter.getStringAttr(tag))});
    };

    auto makeStaticOfr = [&](int64_t v) -> OpFoldResult {
      return rewriter.getIndexAttr(v);
    };

    auto makeRankReducedSubview = [&](Value src, ArrayRef<int64_t> resultShape,
                                      ArrayRef<int64_t> offsets,
                                      ArrayRef<int64_t> sizes,
                                      Location l) -> Value {
      auto srcTy = src.getType().cast<MemRefType>();
      int64_t r = srcTy.getRank();
      SmallVector<OpFoldResult, 4> ofrOffsets;
      SmallVector<OpFoldResult, 4> ofrSizes;
      SmallVector<OpFoldResult, 4> ofrStrides;
      ofrOffsets.reserve(r);
      ofrSizes.reserve(r);
      ofrStrides.reserve(r);
      for (int64_t i = 0; i < r; ++i) {
        ofrOffsets.push_back(makeStaticOfr(offsets[i]));
        ofrSizes.push_back(makeStaticOfr(sizes[i]));
        ofrStrides.push_back(makeStaticOfr(1));
      }
      Type inferred = memref::SubViewOp::inferRankReducedResultType(
          resultShape, srcTy, ofrOffsets, ofrSizes, ofrStrides);
      auto resTy = inferred.cast<MemRefType>();
      return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets, ofrSizes,
                                                ofrStrides);
    };

    auto decodeBatchIndex = [&](int64_t linear) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> idx;
      idx.resize(batchRank, 0);
      for (int64_t d = batchRank; d-- > 0;) {
        int64_t dim = shape[d];
        idx[d] = linear % dim;
        linear /= dim;
      }
      return idx;
    };

    // Fully unrolled: for each batch x M-tile, move -> compute -> move back.
    for (int64_t b = 0; b < batchCount; ++b) {
      SmallVector<int64_t, 4> batchIdx = decodeBatchIndex(b);
      for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
        int64_t mStart = mTile * tileM;
        int64_t mLen = std::min<int64_t>(tileM, M - mStart);
        if (mLen < 1)
          continue;

        int64_t tileCountI64 = ceilDiv(
            static_cast<int64_t>(mLen),
            static_cast<int64_t>(pynq::InstrConfig::kTileSize));
        if (tileCountI64 < 1 || tileCountI64 > pynq::InstrConfig::kMaxTileCount)
          return rewriter.notifyMatchFailure(op, "tile_count out of hardware range [1, 8]");

        int64_t linear = b * numTilesM + mTile;

        SmallVector<int64_t, 8> offsets;
        SmallVector<int64_t, 8> sizes;
        offsets.reserve(rank);
        sizes.reserve(rank);
        for (int64_t d = 0; d < batchRank; ++d) {
          offsets.push_back(batchIdx[d]);
          sizes.push_back(1);
        }
        // axis dimension (rank-2): no tiling, always start at 0.
        offsets.push_back(0);
        offsets.push_back(mStart);
        sizes.push_back(axisLen);
        sizes.push_back(mLen);

        Location copyInLoc = makeTaggedLoc("pynq.int_softmax.copy_in");
        Value inSubview = makeRankReducedSubview(
            input, /*resultShape=*/{axisLen, mLen}, offsets, sizes, copyInLoc);
        rewriter.create<pynq_ops::CopyOp>(copyInLoc, inSubview, bufs[linear]);

        std::string tag = ("pynq.int_softmax.compute.b" + std::to_string(b) +
                           ".m" + std::to_string(mTile));
        Location computeLoc = makeTaggedLoc(tag);
        Value tileCountVal = createI32Const(static_cast<int32_t>(tileCountI64));
        Value reduceKVal = createI32Const(static_cast<int32_t>(axisLen));
        rewriter.create<pynq_ops::SoftmaxOp>(
            computeLoc,
            iscl.cast<TypedValue<MemRefType>>(),
            osclInv.cast<TypedValue<MemRefType>>(),
            bufs[linear],
            tileCountVal,
            reduceKVal);
        rewriter.create<pynq_ops::SyncOp>(makeTaggedLoc("pynq.int_softmax.sync"));

        Location copyOutLoc = makeTaggedLoc("pynq.int_softmax.copy_out");
        Value outSubview = makeRankReducedSubview(
            output, /*resultShape=*/{axisLen, mLen}, offsets, sizes, copyOutLoc);
        rewriter.create<pynq_ops::CopyOp>(copyOutLoc, bufs[linear], outSubview);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.int_layernorm -> pynq.layernorm
//===----------------------------------------------------------------------===//

struct VivadoIntLayerNormToPYNQPattern 
    : public OpRewritePattern<vivado_ops::IntLayerNormOp> {
  using OpRewritePattern<vivado_ops::IntLayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::IntLayerNormOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Location loweredLoc = FusedLoc::get(
        rewriter.getContext(),
        {loc, NameLoc::get(rewriter.getStringAttr(
                  "pynq.lowered_from_vivado.int_layernorm"))});

    // LayerNorm lowering requirements (transpose-mode only):
    // We assume the activation has been transposed such that the reduce axis is
    // the second-to-last dimension, and we only tile along the last dimension.
    bool transposeMode = op.getTransposeMode();
    bool isTransposed = op.getIsTransposed();
    if (!transposeMode)
      return rewriter.notifyMatchFailure(op, "expected transpose_mode=true for PYNQ layernorm");
    if (!isTransposed)
      return rewriter.notifyMatchFailure(op, "expected is_transposed=true for PYNQ layernorm");

    Value output = op.getOutput();
    Value input = op.getInput();
    Value biasInt = op.getBiasInt();
    Value fusedScale = op.getFscl();

    auto inputType = input.getType().dyn_cast<ShapedType>();
    auto outputType = output.getType().dyn_cast<ShapedType>();
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected shaped input/output");

    if (!input.getType().isa<MemRefType>() || !output.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected memref input/output for PYNQ lowering");

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes for PYNQ lowering");

    if (inputType.getRank() < 2 || outputType.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "expected rank >= 2 for layernorm tiling");

    if (inputType.getRank() != outputType.getRank())
      return rewriter.notifyMatchFailure(op, "input/output rank mismatch");
    if (inputType.getShape() != outputType.getShape())
      return rewriter.notifyMatchFailure(op, "expected input/output to have identical shapes");

    auto elemTy = inputType.getElementType();
    if (!elemTy.isInteger(8))
      return rewriter.notifyMatchFailure(op, "expected i8 element type for layernorm lowering");
    if (outputType.getElementType() != elemTy)
      return rewriter.notifyMatchFailure(op, "expected output element type to match input");

    if (!fusedScale || !fusedScale.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected fscl (fused_scale) to be a memref for pynq.layernorm");

    // Bias sanity checks. In PYNQ IR, bias_int is carried as a host-side memref
    // binding (like fused_scale) and is not sliced/partitioned here.
    if (!biasInt)
      return rewriter.notifyMatchFailure(op, "expected bias_int operand");
    if (!biasInt.getType().isa<MemRefType>())
      return rewriter.notifyMatchFailure(op, "expected bias_int to be a memref");
    auto biasType = biasInt.getType().dyn_cast<ShapedType>();
    if (!biasType || !biasType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected bias_int to have a static shaped type");
    if (!biasType.getElementType().isa<IntegerType>())
      return rewriter.notifyMatchFailure(op, "expected bias_int element type to be integer");

    auto shape = inputType.getShape();
    int64_t rank = inputType.getRank();
    int64_t batchRank = rank - 2;
    int64_t reduceLen = shape[rank - 2];
    int64_t M = shape[rank - 1];

    // LayerNorm special-case: broadcast scalar/len-1 fused_scale along channel
    // dimension (reduce axis, rank-2) under transpose-mode.
    if (auto bcast = broadcastLen1ScaleTo(fusedScale, reduceLen, "layernorm_fscl", op, rewriter);
        succeeded(bcast)) {
      fusedScale = *bcast;
    } else {
      return failure();
    }

    if (reduceLen <= 0 || M <= 0)
      return rewriter.notifyMatchFailure(op, "expected non-empty layernorm dimensions");

    // Constraint: reduce axis (rank-2) cannot be split; require reduceLen <= tileN.
    int32_t tileN = pynq::TileConfig::kDefaultTileN;
    if (tileN < 1)
      return rewriter.notifyMatchFailure(op, "invalid tileN configuration");
    if (reduceLen > tileN)
      return rewriter.notifyMatchFailure(op, "layernorm reduce axis exceeds tileN; reduce axis cannot be tiled");
    if (reduceLen > static_cast<int64_t>(pynq::InstrConfig::kMaxReduceK))
      return rewriter.notifyMatchFailure(op, "reduce_k out of hardware range [1, 256]");

    // Bias is expected to align with the reduce axis length.
    auto biasShape = biasType.getShape();
    if (!(biasShape.size() == 1 && biasShape[0] == reduceLen) &&
        !(biasShape.size() == 2 && biasShape[0] == 1 && biasShape[1] == reduceLen)) {
      return rewriter.notifyMatchFailure(op, "expected bias_int shape to match reduce axis (D) length");
    }

    // Tile along the last dimension only.
    int32_t tileM = pynq::TileConfig::kDefaultTileM;
    if (tileM < 1)
      return rewriter.notifyMatchFailure(op, "invalid tileM configuration");

    // Per-tile buffer footprint: reduceLen x tileM (i8).
    int64_t tileBytes = reduceLen * static_cast<int64_t>(tileM) * 1;
    if (tileBytes > pynq::BufferConfig::kDefaultCapacityBytes)
      return rewriter.notifyMatchFailure(op, "tile does not fit in PYNQ buffer capacity");

    auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };
    int64_t numTilesM = ceilDiv(M, tileM);

    int64_t batchCount = 1;
    for (int64_t d = 0; d < batchRank; ++d)
      batchCount *= shape[d];

    int64_t bufCount = std::max<int64_t>(int64_t{1}, batchCount * numTilesM);

    auto i8Type = rewriter.getIntegerType(8);
    auto bufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

    SmallVector<Value, 4> bufs;
    bufs.reserve(bufCount);
    for (int64_t i = 0; i < bufCount; ++i) {
      bufs.push_back(rewriter.create<pynq_ops::BufferAllocOp>(
          loweredLoc, bufferType, rewriter.getStringAttr("activation")));
    }

    auto createI32ConstAt = [&](int32_t value, Location l) {
      return rewriter.create<arith::ConstantOp>(
          l, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
    };
    auto createI32Const = [&](int32_t value) { return createI32ConstAt(value, loweredLoc); };

    auto makeTaggedLoc = [&](StringRef tag) -> Location {
      return FusedLoc::get(
          rewriter.getContext(),
          {loweredLoc, NameLoc::get(rewriter.getStringAttr(tag))});
    };

    auto makeStaticOfr = [&](int64_t v) -> OpFoldResult {
      return rewriter.getIndexAttr(v);
    };

    auto makeRankReducedSubview = [&](Value src, ArrayRef<int64_t> resultShape,
                                      ArrayRef<int64_t> offsets,
                                      ArrayRef<int64_t> sizes,
                                      Location l) -> Value {
      auto srcTy = src.getType().cast<MemRefType>();
      int64_t r = srcTy.getRank();
      SmallVector<OpFoldResult, 4> ofrOffsets;
      SmallVector<OpFoldResult, 4> ofrSizes;
      SmallVector<OpFoldResult, 4> ofrStrides;
      ofrOffsets.reserve(r);
      ofrSizes.reserve(r);
      ofrStrides.reserve(r);
      for (int64_t i = 0; i < r; ++i) {
        ofrOffsets.push_back(makeStaticOfr(offsets[i]));
        ofrSizes.push_back(makeStaticOfr(sizes[i]));
        ofrStrides.push_back(makeStaticOfr(1));
      }
      Type inferred = memref::SubViewOp::inferRankReducedResultType(
          resultShape, srcTy, ofrOffsets, ofrSizes, ofrStrides);
      auto resTy = inferred.cast<MemRefType>();
      return rewriter.create<memref::SubViewOp>(l, resTy, src, ofrOffsets,
                                                ofrSizes, ofrStrides);
    };

    auto decodeBatchIndex = [&](int64_t linear) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> idx;
      idx.resize(batchRank, 0);
      for (int64_t d = batchRank; d-- > 0;) {
        int64_t dim = shape[d];
        idx[d] = linear % dim;
        linear /= dim;
      }
      return idx;
    };

    // Fully unrolled: for each batch x M-tile, move -> compute -> move back.
    for (int64_t b = 0; b < batchCount; ++b) {
      SmallVector<int64_t, 4> batchIdx = decodeBatchIndex(b);
      for (int64_t mTile = 0; mTile < numTilesM; ++mTile) {
        int64_t mStart = mTile * tileM;
        int64_t mLen = std::min<int64_t>(tileM, M - mStart);
        if (mLen < 1)
          continue;

        int64_t tileCountI64 = ceilDiv(
            static_cast<int64_t>(mLen),
            static_cast<int64_t>(pynq::InstrConfig::kTileSize));
        if (tileCountI64 < 1 || tileCountI64 > pynq::InstrConfig::kMaxTileCount)
          return rewriter.notifyMatchFailure(op, "tile_count out of hardware range [1, 8]");

        int64_t linear = b * numTilesM + mTile;

        SmallVector<int64_t, 8> offsets;
        SmallVector<int64_t, 8> sizes;
        offsets.reserve(rank);
        sizes.reserve(rank);
        for (int64_t d = 0; d < batchRank; ++d) {
          offsets.push_back(batchIdx[d]);
          sizes.push_back(1);
        }
        // reduce axis (rank-2): no tiling, always start at 0.
        offsets.push_back(0);
        offsets.push_back(mStart);
        sizes.push_back(reduceLen);
        sizes.push_back(mLen);

        Location copyInLoc = makeTaggedLoc("pynq.int_layernorm.copy_in");
        Value inSubview = makeRankReducedSubview(
            input, /*resultShape=*/{reduceLen, mLen}, offsets, sizes, copyInLoc);
        rewriter.create<pynq_ops::CopyOp>(copyInLoc, inSubview, bufs[linear]);

        std::string tag = ("pynq.int_layernorm.compute.b" + std::to_string(b) +
                           ".m" + std::to_string(mTile));
        Location computeLoc = makeTaggedLoc(tag);
        Value tileCountVal = createI32Const(static_cast<int32_t>(tileCountI64));
        Value reduceKVal = createI32Const(static_cast<int32_t>(reduceLen));
        rewriter.create<pynq_ops::LayerNormOp>(
            computeLoc,
            fusedScale.cast<TypedValue<MemRefType>>(),
          biasInt.cast<TypedValue<MemRefType>>(),
            bufs[linear],
            tileCountVal,
            reduceKVal);
        rewriter.create<pynq_ops::SyncOp>(makeTaggedLoc("pynq.int_layernorm.sync"));

        Location copyOutLoc = makeTaggedLoc("pynq.int_layernorm.copy_out");
        Value outSubview = makeRankReducedSubview(
            output, /*resultShape=*/{reduceLen, mLen}, offsets, sizes, copyOutLoc);
        rewriter.create<pynq_ops::CopyOp>(copyOutLoc, bufs[linear], outSubview);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qconv2d -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQConv2dToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QConv2dOp> {
  using OpRewritePattern<vivado_ops::QConv2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QConv2dOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.qconv2d -> PYNQ lowering
    // Conv2d needs to be transformed to im2col + matmul pattern
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qmatmul_isqrtd -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQMatMulIsqrtDToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QMatMulIsqrtDOp> {
  using OpRewritePattern<vivado_ops::QMatMulIsqrtDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QMatMulIsqrtDOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.qmatmul_isqrtd -> PYNQ lowering
    // Similar to qmatmul but includes 1/sqrt(d) scaling for attention
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.quant -> pynq.quant (CPU-side quantization)
//===----------------------------------------------------------------------===//

struct VivadoQuantToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QuantOp> {
  using OpRewritePattern<vivado_ops::QuantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QuantOp op,
                                 PatternRewriter &rewriter) const override {
    // vivado.quant runs on CPU (PS), not FPGA accelerator (PL)
    // This creates pynq.quant which will be lowered to CPU loop code
    
    Location loc = op.getLoc();
    
    // Extract operands
    Value output = op.getOutput();  // Integer memref
    Value input = op.getInput();    // Float memref
    Value scale = op.getScale();    // Packed i32 scale
    Value zero = op.getZero();      // Optional zero point
    
    // Compose quant_cmb from vivado quant_mode + transpose flags.
    int8_t quantMode = op.getQuantMode();
    auto quantCmb = composeQuantCmbFromVivadoQuantMode(
      quantMode, op.getTransposeMode(), op.getIsTransposed());
    if (failed(quantCmb)) {
      op.emitOpError("invalid quant_mode granularity in vivado.quant; expected bits[2:1] in {00,01,10}");
      return failure();
    }
    
    // Create pynq.quant op (CPU-side operation)
    rewriter.replaceOpWithNewOp<pynq_ops::QuantOp>(
        op, output, input, scale, zero,
      rewriter.getI8IntegerAttr(*quantCmb)
    );
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.dequant -> pynq.dequant (CPU-side dequantization)
//===----------------------------------------------------------------------===//

struct VivadoDequantToPYNQPattern 
    : public OpRewritePattern<vivado_ops::DequantOp> {
  using OpRewritePattern<vivado_ops::DequantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::DequantOp op,
                                 PatternRewriter &rewriter) const override {
    // vivado.dequant runs on CPU (PS), not FPGA accelerator (PL)
    // This creates pynq.dequant which will be lowered to CPU loop code
    
    Location loc = op.getLoc();
    
    // Extract operands
    Value output = op.getOutput();  // Float memref
    Value input = op.getInput();    // Integer memref
    Value scale = op.getScale();    // Packed i32 scale
    Value zero = op.getZero();      // Optional zero point
    
    // Compose quant_cmb from vivado quant_mode + transpose flags.
    int8_t quantMode = op.getQuantMode();
    auto quantCmb = composeQuantCmbFromVivadoQuantMode(
      quantMode, op.getTransposeMode(), op.getIsTransposed());
    if (failed(quantCmb)) {
      op.emitOpError("invalid quant_mode granularity in vivado.dequant; expected bits[2:1] in {00,01,10}");
      return failure();
    }
    
    // Create pynq.dequant op (CPU-side operation)
    rewriter.replaceOpWithNewOp<pynq_ops::DequantOp>(
        op, output, input, scale, zero,
      rewriter.getI8IntegerAttr(*quantCmb)
    );
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.activation_layout_transpose -> pynq.activation_layout_transpose
//===----------------------------------------------------------------------===//

struct VivadoActivationLayoutTransposeToPYNQPattern
    : public OpRewritePattern<vivado_ops::ActivationLayoutTransposeOp> {
  using OpRewritePattern<vivado_ops::ActivationLayoutTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::ActivationLayoutTransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInput().getType().isa<MemRefType>() ||
        !op.getOutput().getType().isa<MemRefType>())
      return failure();

    rewriter.replaceOpWithNewOp<pynq_ops::ActivationLayoutTransposeOp>(
        op, op.getOutput(), op.getInput(), op.getToTransposedAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.activation_dynamic_pad -> pynq.activation_dynamic_pad
//===----------------------------------------------------------------------===//

struct VivadoActivationDynamicPadToPYNQPattern
    : public OpRewritePattern<vivado_ops::ActivationDynamicPadOp> {
  using OpRewritePattern<vivado_ops::ActivationDynamicPadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::ActivationDynamicPadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInput().getType().isa<MemRefType>() ||
        !op.getOutput().getType().isa<MemRefType>())
      return failure();

    rewriter.replaceOpWithNewOp<pynq_ops::ActivationDynamicPadOp>(
        op, op.getOutput(), op.getInput(), op.getToPaddedAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.vit_get_first_token -> pynq.buffer_alloc + pynq.copy + pynq.view + pynq.copy
//===----------------------------------------------------------------------===//

struct VivadoViTGetFirstTokenToPYNQPattern
    : public OpRewritePattern<vivado_ops::ViTGetFirstTokenOp> {
  using OpRewritePattern<vivado_ops::ViTGetFirstTokenOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::ViTGetFirstTokenOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Location loweredLoc = FusedLoc::get(
        rewriter.getContext(),
        {loc, NameLoc::get(rewriter.getStringAttr(
                  "pynq.lowered_from_vivado.vit_get_first_token"))});

    Value output = op.getOutput();
    Value input = op.getInput();

    bool transposeMode = op.getTransposeMode();
    bool isTransposed = transposeMode ? op.getIsTransposed() : false;

    auto inTy = llvm::dyn_cast<MemRefType>(input.getType());
    auto outTy = llvm::dyn_cast<MemRefType>(output.getType());
    if (!inTy || !outTy)
      return rewriter.notifyMatchFailure(op, "expected input/output to be memrefs");
    if (!inTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes for vit_get_first_token lowering");
    if (inTy.getRank() != 3 || outTy.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "expected rank-3 tensors for vit_get_first_token");
    if (inTy.getElementType() != outTy.getElementType())
      return rewriter.notifyMatchFailure(op, "input/output element types must match");
    if (!inTy.getElementType().isInteger(8))
      return rewriter.notifyMatchFailure(op, "only i8 vit_get_first_token lowering is supported");

    ArrayRef<int64_t> inShape = inTy.getShape();
    ArrayRef<int64_t> outShape = outTy.getShape();

    const int64_t B = inShape[0];
    if (B <= 0)
      return rewriter.notifyMatchFailure(op, "expected positive batch dimension");

    int64_t L = 0;
    int64_t D = 0;
    int64_t outTokenDim = 0;
    if (!isTransposed) {
      // input: [B, L, D], output: [B, 1, D]
      // NOTE: In the padded region, VivadoPadding may pad the token dimension
      // (rank-2) from 1 up to tileSize for DMA packing.
      L = inShape[1];
      D = inShape[2];
      int64_t tileSize = static_cast<int64_t>(pynq::InstrConfig::kTileSize);
      outTokenDim = outShape[1];
      if (outShape[0] != B || outShape[2] != D || !(outTokenDim == 1 || outTokenDim == tileSize))
        return rewriter.notifyMatchFailure(
            op, "output shape must be [B,1,D] or [B,tileSize,D] when is_transposed=false");
    } else {
      // input: [B, D, L], output: [B, D, 1]
      // NOTE: A padding pass may run before this lowering and pad the last dim
      // (logical token dimension) from 1 up to tileSize for DMA packing.
      // Therefore, accept output last dimension = 1 or = tileSize.
      D = inShape[1];
      L = inShape[2];
      int64_t tileSize = static_cast<int64_t>(pynq::InstrConfig::kTileSize);
      outTokenDim = outShape[2];
      if (outShape[0] != B || outShape[1] != D || !(outTokenDim == 1 || outTokenDim == tileSize))
        return rewriter.notifyMatchFailure(
            op, "output shape must be [B,D,1] or [B,D,tileSize] when is_transposed=true");
    }

    if (L <= 0 || D <= 0)
      return rewriter.notifyMatchFailure(op, "expected non-empty L/D dimensions");

    // Use default hardware tile sizes for 2D slicing.
    int32_t tileM = pynq::TileConfig::kDefaultTileM;
    int32_t tileN = pynq::TileConfig::kDefaultTileN;
    if (tileM < 1 || tileN < 1)
      return rewriter.notifyMatchFailure(op, "invalid default tileM/tileN configuration");

    // Token axis maps to:
    // - is_transposed=false: L is dimension 1, capped by tileM
    // - is_transposed=true:  L is dimension 2, capped by tileN
    const int32_t tileToken = isTransposed ? tileN : tileM;
    const int32_t tileFeat = isTransposed ? tileM : tileN;

    auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t {
      return (a + b - 1) / b;
    };
    auto alignUp = [&](int64_t a, int64_t b) -> int64_t {
      if (b <= 0)
        return a;
      return ((a + b - 1) / b) * b;
    };

    const int64_t numFeatTiles = ceilDiv(D, tileFeat);
    const int64_t numTokenTiles = ceilDiv(L, tileToken);

    auto i8Type = rewriter.getIntegerType(8);
    auto bufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, pynq::BufferConfig::kDefaultCapacityBytes);

    auto makeStaticOfr = [&](int64_t v) -> OpFoldResult {
      return rewriter.getIndexAttr(v);
    };

    auto makeSubview = [&](Value src, ArrayRef<int64_t> offsets,
                           ArrayRef<int64_t> sizes, Location l) -> Value {
      auto srcTy = llvm::cast<MemRefType>(src.getType());
      SmallVector<OpFoldResult, 4> ofrOffsets;
      SmallVector<OpFoldResult, 4> ofrSizes;
      SmallVector<OpFoldResult, 4> ofrStrides;
      ofrOffsets.reserve(srcTy.getRank());
      ofrSizes.reserve(srcTy.getRank());
      ofrStrides.reserve(srcTy.getRank());
      for (int64_t i = 0, e = srcTy.getRank(); i < e; ++i) {
        ofrOffsets.push_back(makeStaticOfr(offsets[i]));
        ofrSizes.push_back(makeStaticOfr(sizes[i]));
        ofrStrides.push_back(makeStaticOfr(1));
      }
      return rewriter.create<memref::SubViewOp>(l, src, ofrOffsets, ofrSizes, ofrStrides)
          .getResult();
    };

    // We only need the first token (range [0,1)) along L.
    constexpr int64_t viewTokenStart = 0;
    constexpr int64_t viewTokenLen = 1;

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t tokenTile = 0; tokenTile < numTokenTiles; ++tokenTile) {
        int64_t tokenStart = tokenTile * static_cast<int64_t>(tileToken);
        int64_t tokenLen = std::min<int64_t>(tileToken, L - tokenStart);
        if (tokenLen <= 0)
          continue;

        // If this tile doesn't overlap the viewed token range, skip entirely.
        int64_t tileEnd = tokenStart + tokenLen;
        int64_t viewEnd = viewTokenStart + viewTokenLen;
        if (tileEnd <= viewTokenStart || tokenStart >= viewEnd)
          continue;

        // Within an overlapping tile, we always want the first token.
        // IMPORTANT: Which axis corresponds to "row" vs "col" depends on
        // is_transposed because memref layout changes while pynq.view does not
        // permute data. We set splits to describe the logically-valid region.

        for (int64_t featTile = 0; featTile < numFeatTiles; ++featTile) {
          int64_t featStart = featTile * static_cast<int64_t>(tileFeat);
          int64_t featLen = std::min<int64_t>(tileFeat, D - featStart);
          if (featLen <= 0)
            continue;

          int32_t rowSplits = 0;
          int32_t colSplits = 0;
          if (!isTransposed) {
            // Buffer tile is logically [token, feat] (rows=tokens, cols=features).
            rowSplits = 1;
            int64_t colAligned = alignUp(featLen, pynq::InstrConfig::kTileSize);
            colSplits = static_cast<int32_t>(
                std::min<int64_t>(colAligned, static_cast<int64_t>(tileFeat)));
          } else {
            // Buffer tile is logically [feat, token] (rows=features, cols=tokens).
            // Only the first token (col 0) is logically valid; the physical
            // storage may be padded to tileSize, and depad will later slice it.
            rowSplits = static_cast<int32_t>(featLen);
            colSplits = 1;
          }

          // Allocate one virtual on-chip buffer, load a (tokenTile x featTile)
          // slab into it, apply view metadata, and store the first token back.
          Value buf = rewriter.create<pynq_ops::BufferAllocOp>(
              loweredLoc, bufferType, rewriter.getStringAttr("activation"));

          Value inSubview;
          Value outSubview;
          if (!isTransposed) {
            // input [B,L,D], output [B,1,D] (or [B,tileSize,D] after padding)
            inSubview = makeSubview(input, {b, tokenStart, featStart},
                                   {1, tokenLen, featLen}, loweredLoc);
            outSubview = makeSubview(output, {b, 0, featStart},
                                    {1, outTokenDim, featLen}, loweredLoc);
          } else {
            // input [B,D,L], output [B,D,1] (or [B,D,tileSize] after padding)
            inSubview = makeSubview(input, {b, featStart, tokenStart},
                                   {1, featLen, tokenLen}, loweredLoc);
            outSubview = makeSubview(output, {b, featStart, 0},
                                    {1, featLen, outTokenDim}, loweredLoc);
          }

          rewriter.create<pynq_ops::CopyOp>(loweredLoc, inSubview, buf);

          auto viewOp = rewriter.create<pynq_ops::ViewOp>(
              loweredLoc, bufferType, buf,
              rewriter.getI32IntegerAttr(rowSplits),
              rewriter.getI32IntegerAttr(colSplits));

          rewriter.create<pynq_ops::CopyOp>(loweredLoc, viewOp.getOutput(), outSubview);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

bool applyLowerVivadoToPYNQ(ModuleOp &module, MLIRContext *context) {
  // Setup rewrite patterns
  RewritePatternSet patterns(context);
  
  // Add all lowering patterns
  patterns.add<VivadoQMatMulToPYNQPattern>(context);
  patterns.add<VivadoQLinearToPYNQPattern>(context);
  patterns.add<VivadoQAddToPYNQPattern>(context);
  patterns.add<VivadoIntGELUToPYNQPattern>(context);
  patterns.add<VivadoIntSoftmaxToPYNQPattern>(context);
  patterns.add<VivadoIntLayerNormToPYNQPattern>(context);
  patterns.add<VivadoQConv2dToPYNQPattern>(context);
  patterns.add<VivadoQMatMulIsqrtDToPYNQPattern>(context);
  patterns.add<VivadoQuantToPYNQPattern>(context);
  patterns.add<VivadoDequantToPYNQPattern>(context);
  patterns.add<VivadoActivationLayoutTransposeToPYNQPattern>(context);
  patterns.add<VivadoActivationDynamicPadToPYNQPattern>(context);
  patterns.add<VivadoViTGetFirstTokenToPYNQPattern>(context);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return false;

  // Insert pynq.setMagic at the beginning of forward() if allo.hidden_dim exists.
  maybeInsertSetMagic(module);
  return true;
}

struct LowerVivadoToPYNQPass
    : public PassWrapper<LowerVivadoToPYNQPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVivadoToPYNQPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Register PYNQ dialect and other dependencies
    registry.insert<pynq_ops::PYNQDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const final { return "lower-vivado-to-pynq"; }
  
  StringRef getDescription() const final {
    return "Lower Vivado backend ops to PYNQ hardware instructions";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Apply patterns using greedy rewrite
    if (!applyLowerVivadoToPYNQ(module, context)) {
      signalPassFailure();
      return;
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLowerVivadoToPYNQPass() {
  return std::make_unique<LowerVivadoToPYNQPass>();
}

} // namespace allo
} // namespace mlir
