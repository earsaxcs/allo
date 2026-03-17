/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYNQ Mid-level Lowering Pass
 *
 * This pass performs two mid-lowering tasks for the PYNQ backend:
 * 1) Collect per-op bound scale (+bias for layernorm) globals in IR order,
 *    preprocess them (chunking + interleaving), and pack them into per-transfer
 *    memref.global blobs. Insert pynq.data_transfer_instr to stream these blobs
 *    into the hardware scale FIFOs (direction=2/3).
 * 2) Lower compute PYNQ ops to their corresponding *_instr ops, removing the
 *    scale bindings, so that scale transfer can be scheduled independently.
 *
 * Assumptions / Constraints:
 * - Must run after pynq-buffer-allocation; any virtual !pynq.buffer triggers an error.
 * - Function body is expected to be single-block (block-local scheduling model).
 * - Scale/bias operands must be memref.get_global of constant memref.global with
 *   1-D static shape and integer element type.
 * - For matmul and non-layernorm vector ops, scale length must be divisible by
 *   FIFOConfig chunk sizes (32 elements).
 * - If a single op's packed bytes exceed DMAConfig::kMaxPackageCount * 32, the
 *   pass fails (not supported).
 */

#include "PassDetail.h"

#include "allo/Dialect/PYNQConfig.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <optional>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

namespace {

static bool isPynqBuffer(Value v) {
  return v && llvm::isa<pynq::BufferType>(v.getType());
}

static FailureOr<unsigned> getAllocatedBufferId(Value bufferLike,
                                                Operation *forError) {
  if (!bufferLike || !isPynqBuffer(bufferLike))
    return failure();
  auto bt = llvm::cast<pynq::BufferType>(bufferLike.getType());
  if (!bt.isAllocated()) {
    if (forError)
      return forError->emitError() << "PYNQMidLower requires allocated buffers; found virtual !pynq.buffer";
    return failure();
  }
  return bt.getBufferId();
}

struct GlobalIntVector {
  memref::GlobalOp global;
  IntegerType elementType;
  unsigned elementBytes;
  SmallVector<APInt> values;
};

static FailureOr<memref::GlobalOp> resolveGlobalFromMemRefValue(ModuleOp module,
                                                                Value v,
                                                                Operation *user,
                                                                StringRef what) {
  if (!v) {
    if (user)
      user->emitError() << "missing " << what;
    return failure();
  }

  auto getGlobal = v.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal) {
    if (user)
      user->emitError() << what << " must be a memref.get_global of a constant memref.global";
    return failure();
  }

  auto globalOp = module.lookupSymbol<memref::GlobalOp>(getGlobal.getName());
  if (!globalOp) {
    if (user)
      user->emitError() << "failed to resolve memref.global for " << what << ": "
                        << getGlobal.getName();
    return failure();
  }
  return globalOp;
}

static FailureOr<GlobalIntVector>
extract1DIntegerGlobal(ModuleOp module, Value v, Operation *user,
                       StringRef what) {
  auto globalOpOr = resolveGlobalFromMemRefValue(module, v, user, what);
  if (failed(globalOpOr))
    return failure();
  memref::GlobalOp globalOp = *globalOpOr;

  auto memrefTy = globalOp.getType().dyn_cast<MemRefType>();
  if (!memrefTy || !memrefTy.hasStaticShape() || memrefTy.getRank() != 1) {
    if (user)
      user->emitError() << what << " must be a 1-D static memref.global";
    return failure();
  }

  auto elemTy = memrefTy.getElementType().dyn_cast<IntegerType>();
  if (!elemTy) {
    if (user)
      user->emitError() << what << " must have integer element type";
    return failure();
  }
  if (elemTy.getWidth() > 64) {
    if (user)
      user->emitError() << what << " element width > 64 is not supported";
    return failure();
  }

  auto init = globalOp.getInitialValue();
  if (!init) {
    if (user)
      user->emitError() << what << " global has no initial value";
    return failure();
  }

  auto dense = init->dyn_cast<DenseElementsAttr>();
  if (!dense) {
    if (user)
      user->emitError() << what << " global initial value is not DenseElementsAttr";
    return failure();
  }

  // Ensure element type matches; DenseElementsAttr is a tensor, but should match.
  auto denseElemTy = dense.getElementType().dyn_cast<IntegerType>();
  if (!denseElemTy || denseElemTy.getWidth() != elemTy.getWidth()) {
    if (user)
      user->emitError() << what << " DenseElementsAttr element type mismatch";
    return failure();
  }

  GlobalIntVector out;
  out.global = globalOp;
  out.elementType = elemTy;
  out.elementBytes = (elemTy.getWidth() + 7) / 8;

  out.values.reserve(dense.getNumElements());
  for (const APInt &x : dense.getValues<APInt>())
    out.values.push_back(x);

  // Sanity: shape size.
  int64_t n = memrefTy.getShape()[0];
  if (n >= 0 && static_cast<int64_t>(out.values.size()) != n) {
    if (user)
      user->emitError() << what << " element count mismatch with memref shape";
    return failure();
  }

  return out;
}

static void appendElementBytesLittleEndian(SmallVectorImpl<uint8_t> &out,
                                           const APInt &v,
                                           unsigned elemBytes) {
  // APInt stores bits; serialize little-endian.
  // Works for <=64-bit values.
  uint64_t raw = 0;
  if (v.getBitWidth() <= 64)
    raw = v.getZExtValue();
  for (unsigned i = 0; i < elemBytes; ++i)
    out.push_back(static_cast<uint8_t>((raw >> (8 * i)) & 0xFFu));
}

static void debugDumpBytesHex(llvm::raw_ostream &os, ArrayRef<uint8_t> bytes,
                              size_t bytesPerLine = 32) {
  os << "bytes(" << bytes.size() << ") = [";
  if (bytes.empty()) {
    os << "]\n";
    return;
  }
  os << "\n";
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i % bytesPerLine == 0)
      os << "  " << llvm::formatv("{0,6}:", i);
    os << " " << llvm::formatv("0x{0:X2}", bytes[i]);
    if (i % bytesPerLine == bytesPerLine - 1 || i + 1 == bytes.size())
      os << "\n";
  }
  os << "]\n";
}

static void debugDumpAPIntVector(llvm::raw_ostream &os,
                                 ArrayRef<APInt> values,
                                 unsigned elemBytes,
                                 size_t elemsPerLine = 8) {
  os << "values(" << values.size() << ") elemBytes=" << elemBytes << " = [\n";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i % elemsPerLine == 0)
      os << "  " << llvm::formatv("{0,6}:", i);
    uint64_t raw = 0;
    if (values[i].getBitWidth() <= 64)
      raw = values[i].getZExtValue();
    os << " " << llvm::formatv("0x{0:X}", raw);
    if (i % elemsPerLine == elemsPerLine - 1 || i + 1 == values.size())
      os << "\n";
  }
  os << "]\n";
}

static std::optional<int64_t> tryGetConstI64(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
    return static_cast<int64_t>(c.value());
  return std::nullopt;
}

static void debugPrintValueOrConst(llvm::raw_ostream &os, Value v) {
  if (!v) {
    os << "<null>";
    return;
  }
  if (auto c = tryGetConstI64(v)) {
    os << *c;
    return;
  }
  os << "<";
  v.print(os);
  os << ">";
}

static FailureOr<SmallVector<uint8_t>>
packChunkHalfInterleaveBytes(ArrayRef<APInt> values,
                             unsigned elemBytes,
                             int32_t chunkElements,
                             Operation *user,
                             StringRef what,
                             bool debugScalePack) {
  if (chunkElements <= 0) {
    if (user)
      user->emitError() << "invalid chunk size";
    return failure();
  }
  if (static_cast<int64_t>(values.size()) % chunkElements != 0) {
    if (user)
      user->emitError() << what << " length " << values.size()
                        << " is not divisible by chunk size " << chunkElements;
    return failure();
  }
  if (chunkElements % 2 != 0) {
    if (user)
      user->emitError() << "chunk size must be even for half-interleave";
    return failure();
  }

  SmallVector<uint8_t> out;
  out.reserve(values.size() * elemBytes);

  if (debugScalePack) {
    llvm::errs() << "[pynq-mid-lower][pack] " << what << "\n";
    llvm::errs() << "  chunkElements=" << chunkElements
                 << " half=" << (chunkElements / 2)
                 << " elemBytes=" << elemBytes << "\n";
    debugDumpAPIntVector(llvm::errs(), values, elemBytes);
  }

  int32_t half = chunkElements / 2;
  for (size_t base = 0; base < values.size(); base += chunkElements) {
    if (debugScalePack) {
      llvm::errs() << "  chunk base=" << base << ".." << (base + chunkElements)
                   << " (half-interleave pairs: i and i+" << half << ")\n";
    }
    for (int32_t i = 0; i < half; ++i) {
      if (debugScalePack) {
        uint64_t a = values[base + i].getBitWidth() <= 64
                         ? values[base + i].getZExtValue()
                         : 0;
        uint64_t b = values[base + i + half].getBitWidth() <= 64
                         ? values[base + i + half].getZExtValue()
                         : 0;
        llvm::errs() << "    pair (" << (base + i) << "," << (base + i + half)
                     << ") values=(" << llvm::formatv("0x{0:X}", a) << ","
                     << llvm::formatv("0x{0:X}", b) << ")\n";
      }
      appendElementBytesLittleEndian(out, values[base + i], elemBytes);
      appendElementBytesLittleEndian(out, values[base + i + half], elemBytes);
    }
  }

  if (debugScalePack) {
    llvm::errs() << "  packed (after half-interleave):\n";
    debugDumpBytesHex(llvm::errs(), out);
  }

  return out;
}

static FailureOr<SmallVector<uint8_t>>
packMatMulScale(ModuleOp module, pynq::MatMulOp op, bool debugScalePack) {
  auto gvOr = extract1DIntegerGlobal(module, op.getFusedScale(), op, "fused_scale");
  if (failed(gvOr))
    return failure();
  auto &gv = *gvOr;

  if (debugScalePack) {
    llvm::errs() << "[pynq-mid-lower][pack] matmul fused_scale global=@"
                 << gv.global.getName() << " len=" << gv.values.size()
                 << " elemBytes=" << gv.elementBytes << "\n";
  }

  return packChunkHalfInterleaveBytes(gv.values, gv.elementBytes,
                                      FIFOConfig::kMatrixScaleChunk,
                                      op, "matmul fused_scale",
                                      debugScalePack);
}

static FailureOr<SmallVector<uint8_t>>
packVectorScalesStacked(ModuleOp module,
                        ArrayRef<std::pair<Value, StringRef>> scales,
                        Operation *op,
                        int32_t chunkElements,
                        bool debugScalePack) {
  if (scales.empty())
    return SmallVector<uint8_t>();

  SmallVector<GlobalIntVector> vecs;
  vecs.reserve(scales.size());

  for (auto it : scales) {
    auto gvOr = extract1DIntegerGlobal(module, it.first, op, it.second);
    if (failed(gvOr))
      return failure();
    vecs.push_back(std::move(*gvOr));
  }

  // All vector scales must share element width for stacked packing.
  unsigned elemBytes = vecs.front().elementBytes;
  for (auto &v : vecs) {
    if (v.elementBytes != elemBytes) {
      op->emitError() << "vector scale element sizes mismatch; expected "
                      << elemBytes << " bytes";
      return failure();
    }
  }

  // Allow per-tensor (length-1) scales to broadcast to the common length.
  size_t len = 0;
  for (auto &v : vecs)
    len = std::max(len, v.values.size());
  if (len == 0) {
    op->emitError() << "vector scale length is zero";
    return failure();
  }

  for (auto &v : vecs) {
    if (v.values.size() == len)
      continue;
    if (v.values.size() == 1)
      continue;
    op->emitError() << "vector scale lengths mismatch";
    return failure();
  }

  if (static_cast<int64_t>(len) % chunkElements != 0) {
    op->emitError() << "vector scale length " << len
                    << " is not divisible by chunk size " << chunkElements;
    return failure();
  }

  size_t blocks = len / chunkElements;
  SmallVector<uint8_t> out;
  out.reserve(len * elemBytes * vecs.size());

  if (debugScalePack) {
    llvm::errs() << "[pynq-mid-lower][pack] vector scales stacked\n";
    llvm::errs() << "  scales=" << vecs.size() << " len=" << len
                 << " chunkElements=" << chunkElements
                 << " blocks=" << blocks << " elemBytes=" << elemBytes << "\n";
    for (size_t si = 0; si < vecs.size(); ++si) {
      llvm::errs() << "  scale[" << si << "] " << scales[si].second << ": ";
      llvm::errs() << "global=@" << vecs[si].global.getName() << " ";
      llvm::errs() << "rawLen=" << vecs[si].values.size();
      if (vecs[si].values.size() == 1 && len > 1)
        llvm::errs() << " (broadcast)";
      llvm::errs() << "\n";
      debugDumpAPIntVector(llvm::errs(), vecs[si].values, vecs[si].elementBytes);
    }
  }

  int32_t half = chunkElements / 2;
  // Emit packed bytes in chunk-major order across scale streams:
  // [scale0_chunk0, scale1_chunk0, ..., scaleN_chunk0, scale0_chunk1, ...].
  for (size_t b = 0; b < blocks; ++b) {
    size_t base = b * chunkElements;
    for (auto &s : vecs) {
      auto getVal = [&](size_t idx) -> const APInt & {
        return s.values.size() == 1 ? s.values[0] : s.values[idx];
      };
      for (int32_t i = 0; i < half; ++i) {
        if (debugScalePack) {
          uint64_t a = getVal(base + i).getBitWidth() <= 64
                           ? getVal(base + i).getZExtValue()
                           : 0;
          uint64_t bval = getVal(base + i + half).getBitWidth() <= 64
                              ? getVal(base + i + half).getZExtValue()
                              : 0;
          llvm::errs() << "  block=" << b << " scale=@" << s.global.getName()
                       << " pair(" << (base + i) << "," << (base + i + half)
                       << ") values=(" << llvm::formatv("0x{0:X}", a) << ","
                       << llvm::formatv("0x{0:X}", bval) << ")\n";
        }
        appendElementBytesLittleEndian(out, getVal(base + i), elemBytes);
        appendElementBytesLittleEndian(out, getVal(base + i + half), elemBytes);
      }
    }
  }

  if (debugScalePack) {
    llvm::errs() << "  packed (after stacking + half-interleave):\n";
    debugDumpBytesHex(llvm::errs(), out);
  }

  return out;
}

static FailureOr<SmallVector<uint8_t>>
packGELUScale(ModuleOp module, pynq::GELUOp op, bool debugScalePack) {
  return packVectorScalesStacked(module,
                                 {{op.getInScale(), "in_scale"},
                                  {op.getOutScaleInv(), "out_scale_inv"}},
                                 op,
                                 FIFOConfig::kVectorScaleChunk,
                                 debugScalePack);
}

static FailureOr<SmallVector<uint8_t>>
packSoftmaxScale(ModuleOp module, pynq::SoftmaxOp op, bool debugScalePack) {
  return packVectorScalesStacked(module,
                                 {{op.getInScale(), "in_scale"},
                                  {op.getOutScaleInv(), "out_scale_inv"}},
                                 op,
                                 FIFOConfig::kVectorScaleChunk,
                                 debugScalePack);
}

static FailureOr<SmallVector<uint8_t>>
packQAddScale(ModuleOp module, pynq::QAddOp op, bool debugScalePack) {
  // Block-stacking order: y, x, o_inv.
  return packVectorScalesStacked(module,
                                 {{op.getYScale(), "y_scale"},
                                  {op.getXScale(), "x_scale"},
                                  {op.getOScaleInv(), "o_scale_inv"}},
                                 op,
                                 FIFOConfig::kVectorScaleChunk,
                                 debugScalePack);
}

static FailureOr<SmallVector<uint8_t>>
packLayerNormScaleBias(ModuleOp module,
                       pynq::LayerNormOp op,
                       bool debugScalePack) {
  auto scaleOr = extract1DIntegerGlobal(module, op.getFusedScale(), op, "fused_scale");
  if (failed(scaleOr))
    return failure();
  auto biasOr = extract1DIntegerGlobal(module, op.getBiasInt(), op, "bias_int");
  if (failed(biasOr))
    return failure();

  auto &s = *scaleOr;
  auto &b = *biasOr;

  if (s.values.size() != b.values.size()) {
    op.emitError() << "layernorm scale and bias lengths mismatch";
    return failure();
  }

  SmallVector<uint8_t> out;
  out.reserve(s.values.size() * (s.elementBytes + b.elementBytes));
  for (size_t i = 0; i < s.values.size(); ++i) {
    appendElementBytesLittleEndian(out, b.values[i], b.elementBytes);
    appendElementBytesLittleEndian(out, s.values[i], s.elementBytes);
  }

  if (debugScalePack) {
    llvm::errs() << "[pynq-mid-lower][pack] layernorm fused_scale+bias_int\n";
    llvm::errs() << "  bias_int global=@" << b.global.getName()
                 << " len=" << b.values.size()
                 << " elemBytes=" << b.elementBytes << "\n";
    debugDumpAPIntVector(llvm::errs(), b.values, b.elementBytes);
    llvm::errs() << "  fused_scale global=@" << s.global.getName()
                 << " len=" << s.values.size()
                 << " elemBytes=" << s.elementBytes << "\n";
    debugDumpAPIntVector(llvm::errs(), s.values, s.elementBytes);
    llvm::errs() << "  packed (bias then scale, per element):\n";
    debugDumpBytesHex(llvm::errs(), out);
  }

  return out;
}

static void padToPackageMultiple(SmallVectorImpl<uint8_t> &bytes) {
  int32_t pkg = DMAConfig::kBytesPerPackage;
  if (pkg <= 0)
    return;
  size_t rem = bytes.size() % static_cast<size_t>(pkg);
  if (rem == 0)
    return;
  size_t need = static_cast<size_t>(pkg) - rem;
  bytes.append(need, 0);
}

struct PackedOpData {
  Operation *op = nullptr;          // anchor op in IR order
  SmallVector<uint8_t> bytes;       // packed payload bytes for this op
};

struct TransferSegment {
  Operation *insertBefore = nullptr; // insertion point in block
  SmallVector<uint8_t> bytes;        // concatenated (and padded) payload
  int32_t direction = 0;             // 2=matrix fifo, 3=vector fifo
};

static FailureOr<SmallVector<TransferSegment>>
makeSegments(ArrayRef<PackedOpData> ops, int32_t direction, int64_t maxBytes,
             Operation *forError) {
  SmallVector<TransferSegment> segments;

  TransferSegment current;
  current.direction = direction;

  int64_t curBytes = 0;
  for (auto &pod : ops) {
    int64_t opBytes = static_cast<int64_t>(pod.bytes.size());
    if (opBytes > maxBytes) {
      if (forError)
        forError->emitError() << "single op packed scale size " << opBytes
                              << " exceeds max transfer bytes " << maxBytes;
      return failure();
    }

    if (curBytes == 0) {
      current.insertBefore = pod.op;
      current.bytes.clear();
      current.bytes.append(pod.bytes.begin(), pod.bytes.end());
      curBytes = opBytes;
      continue;
    }

    if (curBytes + opBytes > maxBytes) {
      padToPackageMultiple(current.bytes);
      segments.push_back(std::move(current));
      current = TransferSegment();
      current.direction = direction;
      current.insertBefore = pod.op;
      current.bytes.append(pod.bytes.begin(), pod.bytes.end());
      curBytes = opBytes;
      continue;
    }

    current.bytes.append(pod.bytes.begin(), pod.bytes.end());
    curBytes += opBytes;
  }

  if (curBytes > 0) {
    padToPackageMultiple(current.bytes);
    segments.push_back(std::move(current));
  }

  return segments;
}

static FailureOr<int64_t> getStaticMemRefByteSize(Value memrefVal,
                                                  Operation *user,
                                                  StringRef what) {
  auto ty = memrefVal.getType().dyn_cast<MemRefType>();
  if (!ty || !ty.hasStaticShape()) {
    user->emitError() << what << " must be a static memref";
    return failure();
  }
  auto elemTy = ty.getElementType();
  auto intTy = elemTy.dyn_cast<IntegerType>();
  auto floatTy = elemTy.dyn_cast<FloatType>();
  unsigned elemBits = 0;
  if (intTy)
    elemBits = intTy.getWidth();
  else if (floatTy)
    elemBits = floatTy.getWidth();
  else {
    user->emitError() << what << " element type unsupported for size calc";
    return failure();
  }
  if (elemBits % 8 != 0) {
    user->emitError() << what << " element bitwidth not byte-aligned";
    return failure();
  }
  int64_t elemBytes = elemBits / 8;

  int64_t elements = 1;
  for (int64_t d : ty.getShape()) {
    if (d < 0) {
      user->emitError() << what << " has dynamic dimension";
      return failure();
    }
    elements *= d;
  }

  return elements * elemBytes;
}

static FailureOr<int32_t> computeTileCountFromLastDim(Value memrefVal,
                                                      Operation *user) {
  auto ty = memrefVal.getType().dyn_cast<MemRefType>();
  if (!ty || !ty.hasStaticShape() || ty.getRank() < 1) {
    user->emitError() << "memref must be static rank>=1 for tile_count";
    return failure();
  }

  int64_t lastDim = ty.getShape().back();
  if (lastDim < 0) {
    user->emitError() << "last dim must be static for tile_count";
    return failure();
  }

  auto elemTy = ty.getElementType();
  unsigned elemBits = 0;
  if (auto intTy = elemTy.dyn_cast<IntegerType>())
    elemBits = intTy.getWidth();
  else if (auto fTy = elemTy.dyn_cast<FloatType>())
    elemBits = fTy.getWidth();
  else {
    user->emitError() << "unsupported element type for tile_count";
    return failure();
  }
  if (elemBits % 8 != 0) {
    user->emitError() << "element type not byte-aligned for tile_count";
    return failure();
  }

  int64_t lastBytes = lastDim * static_cast<int64_t>(elemBits / 8);
  if (lastBytes % InstrConfig::kTileSize != 0) {
    user->emitError() << "last-dim byte size " << lastBytes
                      << " not divisible by InstrConfig::kTileSize="
                      << InstrConfig::kTileSize;
    return failure();
  }

  int64_t tiles = lastBytes / InstrConfig::kTileSize;
  if (tiles < 1 || tiles > static_cast<int64_t>(InstrConfig::kMaxTileCount)) {
    user->emitError() << "tile_count out of range: " << tiles;
    return failure();
  }
  return static_cast<int32_t>(tiles);
}

class PYNQMidLowerPass
    : public PassWrapper<PYNQMidLowerPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQMidLowerPass)

  PYNQMidLowerPass() = default;
  PYNQMidLowerPass(const PYNQMidLowerPass &pass) : PassWrapper(pass) {}

  Option<bool> debugScalePack{
    *this, "debug-scale-pack",
    llvm::cl::desc(
      "Enable verbose debug printing for scale/bias packing (raw values, intermediate pairing, packed bytes, transfer segments)"),
    llvm::cl::init(false)};

  Option<bool> debugOpConvert{
    *this, "debug-op-convert",
    llvm::cl::desc(
      "Enable debug printing for op conversion to *_instr (buffer ids, tile counts, and lowered instruction fields)"),
    llvm::cl::init(false)};

  StringRef getArgument() const final { return "pynq-mid-lower"; }
  StringRef getDescription() const final {
    return "Mid-lower PYNQ: pack scale/bias to FIFO transfers and convert ops to instr";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void PYNQMidLowerPass::runOnOperation() {
  ModuleOp module = getOperation();

  int64_t maxTransferBytes =
      static_cast<int64_t>(DMAConfig::kMaxPackageCount) *
      static_cast<int64_t>(DMAConfig::kBytesPerPackage);

  bool failedAny = false;

  module.walk([&](func::FuncOp funcOp) {
    if (failedAny)
      return;

    if (debugScalePack || debugOpConvert) {
      llvm::errs() << "[pynq-mid-lower] func @" << funcOp.getName() << "\n";
      if (debugScalePack)
        llvm::errs() << "  debug-scale-pack=1\n";
      if (debugOpConvert)
        llvm::errs() << "  debug-op-convert=1\n";
    }

    // Single-block assumption.
    if (!funcOp.getBody().hasOneBlock()) {
      funcOp.emitError() << "PYNQMidLower expects single-block functions";
      failedAny = true;
      return;
    }

    Block &block = funcOp.getBody().front();

    // First pass: collect packed per-op data.
    SmallVector<PackedOpData> matrixOps;
    SmallVector<PackedOpData> vectorOps;

    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      // Validate that any buffer operands are allocated.
      for (Value operand : op.getOperands()) {
        if (!isPynqBuffer(operand))
          continue;
        if (failed(getAllocatedBufferId(operand, &op))) {
          failedAny = true;
          return;
        }
      }

      if (auto mm = llvm::dyn_cast<pynq::MatMulOp>(&op)) {
        auto bytesOr = packMatMulScale(module, mm, debugScalePack);
        if (failed(bytesOr)) {
          failedAny = true;
          return;
        }
        if (debugScalePack)
          llvm::errs() << "[pynq-mid-lower][pack] MatMulOp packedBytes="
                       << bytesOr->size() << "\n";
        matrixOps.push_back(PackedOpData{&op, std::move(*bytesOr)});
        continue;
      }

      if (auto gelu = llvm::dyn_cast<pynq::GELUOp>(&op)) {
        auto bytesOr = packGELUScale(module, gelu, debugScalePack);
        if (failed(bytesOr)) {
          failedAny = true;
          return;
        }
        if (debugScalePack)
          llvm::errs() << "[pynq-mid-lower][pack] GELUOp packedBytes="
                       << bytesOr->size() << "\n";
        vectorOps.push_back(PackedOpData{&op, std::move(*bytesOr)});
        continue;
      }

      if (auto sm = llvm::dyn_cast<pynq::SoftmaxOp>(&op)) {
        auto bytesOr = packSoftmaxScale(module, sm, debugScalePack);
        if (failed(bytesOr)) {
          failedAny = true;
          return;
        }
        if (debugScalePack)
          llvm::errs() << "[pynq-mid-lower][pack] SoftmaxOp packedBytes="
                       << bytesOr->size() << "\n";
        vectorOps.push_back(PackedOpData{&op, std::move(*bytesOr)});
        continue;
      }

      if (auto qa = llvm::dyn_cast<pynq::QAddOp>(&op)) {
        auto bytesOr = packQAddScale(module, qa, debugScalePack);
        if (failed(bytesOr)) {
          failedAny = true;
          return;
        }
        if (debugScalePack)
          llvm::errs() << "[pynq-mid-lower][pack] QAddOp packedBytes="
                       << bytesOr->size() << "\n";
        vectorOps.push_back(PackedOpData{&op, std::move(*bytesOr)});
        continue;
      }

      if (auto ln = llvm::dyn_cast<pynq::LayerNormOp>(&op)) {
        auto bytesOr = packLayerNormScaleBias(module, ln, debugScalePack);
        if (failed(bytesOr)) {
          failedAny = true;
          return;
        }
        if (debugScalePack)
          llvm::errs() << "[pynq-mid-lower][pack] LayerNormOp packedBytes="
                       << bytesOr->size() << "\n";
        vectorOps.push_back(PackedOpData{&op, std::move(*bytesOr)});
        continue;
      }
    }

    if (debugScalePack) {
      llvm::errs() << "[pynq-mid-lower][pack] collected matrixOps="
                   << matrixOps.size() << " vectorOps=" << vectorOps.size()
                   << "\n";
    }

    // Build transfer segments (per FIFO).
    auto matrixSegOr = makeSegments(matrixOps, FIFOConfig::kDirectionMatrixScaleFIFO,
                                    maxTransferBytes, funcOp);
    if (failed(matrixSegOr)) {
      failedAny = true;
      return;
    }

    auto vectorSegOr = makeSegments(vectorOps, FIFOConfig::kDirectionVectorScaleFIFO,
                                    maxTransferBytes, funcOp);
    if (failed(vectorSegOr)) {
      failedAny = true;
      return;
    }

    if (debugScalePack) {
      auto dumpSegs = [&](ArrayRef<TransferSegment> segs, StringRef kind) {
        llvm::errs() << "[pynq-mid-lower][pack] segments kind=" << kind
                     << " count=" << segs.size() << "\n";
        for (size_t i = 0; i < segs.size(); ++i) {
          const auto &seg = segs[i];
          llvm::errs() << "  seg[" << i << "] direction=" << seg.direction
                       << " bytes=" << seg.bytes.size();
          if (seg.insertBefore) {
            llvm::errs() << " insertBefore=" << seg.insertBefore->getName();
          }
          llvm::errs() << "\n";
          debugDumpBytesHex(llvm::errs(), seg.bytes);
        }
      };
      dumpSegs(*matrixSegOr, "matrix");
      dumpSegs(*vectorSegOr, "vector");
    }

    // Insert new globals + transfer instrs.
    OpBuilder moduleBuilder(module.getContext());
    moduleBuilder.setInsertionPointToStart(module.getBody());

    auto buildI32Const = [&](OpBuilder &b, Location loc, int32_t v) -> Value {
      return b.create<arith::ConstantIntOp>(loc, v, 32);
    };

    auto createPackedGlobalI8 = [&](StringRef symName,
                                   ArrayRef<uint8_t> bytes,
                                   Location loc) -> memref::GlobalOp {
      auto i8 = moduleBuilder.getIntegerType(8);
      auto memrefTy = MemRefType::get({static_cast<int64_t>(bytes.size())}, i8);

      // DenseElementsAttr expects a tensor type.
      SmallVector<int8_t> signedBytes;
      signedBytes.reserve(bytes.size());
      for (uint8_t b : bytes)
        signedBytes.push_back(static_cast<int8_t>(b));

      auto initAttr = DenseElementsAttr::get(
          RankedTensorType::get(memrefTy.getShape(), i8),
          llvm::ArrayRef(signedBytes));

      return moduleBuilder.create<memref::GlobalOp>(
          loc, symName, moduleBuilder.getStringAttr("private"), memrefTy,
          initAttr, /*constant=*/true, /*alignment=*/nullptr);
    };

    // NOTE: scale-pack transfers are inserted without any pynq.sync.
    // They stream constant payload to FIFOs and should not introduce extra waits.
    auto insertScalePackTransfersWithoutSync = [&](ArrayRef<TransferSegment> segs,
                                                   StringRef kindPrefix) {
      int idx = 0;
      for (auto &seg : segs) {
        if (!seg.insertBefore)
          continue;
        std::string sym =
            llvm::formatv("__pynq_{0}_{1}_{2}", funcOp.getName(), kindPrefix, idx)
                .str();
        ++idx;

        auto global = createPackedGlobalI8(sym, seg.bytes, funcOp.getLoc());

        if (debugScalePack) {
          llvm::errs() << "[pynq-mid-lower][pack] create global @" << sym
                       << " bytes=" << seg.bytes.size()
                       << " direction=" << seg.direction << "\n";
        }

        OpBuilder b(seg.insertBefore);
        auto memrefTy = global.getType();
        auto getg = b.create<memref::GetGlobalOp>(seg.insertBefore->getLoc(),
                                                  memrefTy, global.getName());

        int32_t pkgNum = static_cast<int32_t>(seg.bytes.size() /
                                              static_cast<size_t>(DMAConfig::kBytesPerPackage));

        auto bufferId = buildI32Const(b, seg.insertBefore->getLoc(), 0);
        auto tileCount = buildI32Const(b, seg.insertBefore->getLoc(), 0);
        auto totalPkg = buildI32Const(b, seg.insertBefore->getLoc(), pkgNum);
        auto dir = buildI32Const(b, seg.insertBefore->getLoc(), seg.direction);

        b.create<pynq::DataTransferInstrOp>(seg.insertBefore->getLoc(),
                                            getg.getResult(), bufferId,
                                            tileCount, totalPkg, dir);

        if (debugScalePack) {
          llvm::errs() << "[pynq-mid-lower][pack] insert DataTransferInstrOp kind="
                       << kindPrefix << " pkgNum=" << pkgNum
                       << " dir=" << seg.direction << "\n";
        }
      }
    };

    insertScalePackTransfersWithoutSync(*matrixSegOr, "matrix_scale_pack");
    insertScalePackTransfersWithoutSync(*vectorSegOr, "vector_scale_pack");

    // Second pass: rewrite ops to instr.
    SmallVector<Operation *> toErase;
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      OpBuilder b(&op);
      Location loc = op.getLoc();

      auto i32c = [&](int32_t v) -> Value {
        return b.create<arith::ConstantIntOp>(loc, v, 32);
      };

      if (auto mm = llvm::dyn_cast<pynq::MatMulOp>(&op)) {
        auto inIdOr = getAllocatedBufferId(mm.getInputBuffer(), &op);
        auto wIdOr = getAllocatedBufferId(mm.getWeightBuffer(), &op);
        auto oIdOr = getAllocatedBufferId(mm.getOutputBuffer(), &op);
        if (failed(inIdOr) || failed(wIdOr) || failed(oIdOr)) {
          failedAny = true;
          return;
        }
        if (debugOpConvert) {
          llvm::errs() << "[pynq-mid-lower][convert] MatMulOp ";
          loc.print(llvm::errs());
          llvm::errs() << " in=" << *inIdOr << " w=" << *wIdOr
                       << " out=" << *oIdOr << " input_tile_count=";
          debugPrintValueOrConst(llvm::errs(), mm.getInputTileCount());
          llvm::errs() << " weight_tile_count=";
          debugPrintValueOrConst(llvm::errs(), mm.getWeightTileCount());
          llvm::errs() << " reduceK=";
          debugPrintValueOrConst(llvm::errs(), mm.getReduceK());
          llvm::errs() << " headIdx=";
          debugPrintValueOrConst(llvm::errs(), mm.getHeadIdx());
          llvm::errs() << " headTileAxis=";
          debugPrintValueOrConst(llvm::errs(), mm.getHeadTileAxis());
          llvm::errs() << " enableBias=";
          debugPrintValueOrConst(llvm::errs(), mm.getEnableBias());
          llvm::errs() << " enableTranspose=";
          debugPrintValueOrConst(llvm::errs(), mm.getEnableTranspose());
          llvm::errs() << "\n";
        }
        b.create<pynq::MatMulInstrOp>(
            loc,
            i32c(static_cast<int32_t>(*inIdOr)),
            i32c(static_cast<int32_t>(*wIdOr)),
            i32c(static_cast<int32_t>(*oIdOr)),
            mm.getInputTileCount(), mm.getWeightTileCount(), mm.getReduceK(),
            mm.getHeadIdx(), mm.getHeadTileAxis(), mm.getEnableBias(),
            mm.getEnableTranspose());
        toErase.push_back(&op);
        continue;
      }

      auto lowerSimpleVector = [&](auto concreteOp, int32_t opCode) {
        auto bufIdOr = getAllocatedBufferId(concreteOp.getBuffer(), &op);
        if (failed(bufIdOr)) {
          failedAny = true;
          return;
        }
        if (debugOpConvert) {
          llvm::errs() << "[pynq-mid-lower][convert] VectorOp code=" << opCode
                       << " ";
          op.getLoc().print(llvm::errs());
          llvm::errs() << " buffer=" << *bufIdOr << " tile_count=";
          debugPrintValueOrConst(llvm::errs(), concreteOp.getTileCount());
          llvm::errs() << " reduceK=";
          debugPrintValueOrConst(llvm::errs(), concreteOp.getReduceK());
          llvm::errs() << "\n";
        }
        b.create<pynq::VectorInstrOp>(
            loc, i32c(static_cast<int32_t>(*bufIdOr)),
            concreteOp.getTileCount(), concreteOp.getReduceK(), i32c(opCode),
            i32c(0));
        toErase.push_back(&op);
      };

      if (auto gelu = llvm::dyn_cast<pynq::GELUOp>(&op)) {
        lowerSimpleVector(gelu, static_cast<int32_t>(pynq::VectorOpCode::kGELU));
        continue;
      }
      if (auto sm = llvm::dyn_cast<pynq::SoftmaxOp>(&op)) {
        lowerSimpleVector(sm, static_cast<int32_t>(pynq::VectorOpCode::kSoftmax));
        continue;
      }
      if (auto ln = llvm::dyn_cast<pynq::LayerNormOp>(&op)) {
        lowerSimpleVector(ln, static_cast<int32_t>(pynq::VectorOpCode::kLayerNorm));
        continue;
      }

      if (auto qa = llvm::dyn_cast<pynq::QAddOp>(&op)) {
        auto bufIdOr = getAllocatedBufferId(qa.getBuffer(), &op);
        auto extraIdOr = getAllocatedBufferId(qa.getExtraBuffer(), &op);
        if (failed(bufIdOr) || failed(extraIdOr)) {
          failedAny = true;
          return;
        }
        if (debugOpConvert) {
          llvm::errs() << "[pynq-mid-lower][convert] QAddOp ";
          loc.print(llvm::errs());
          llvm::errs() << " buffer=" << *bufIdOr
                       << " extra=" << *extraIdOr << " tile_count=";
          debugPrintValueOrConst(llvm::errs(), qa.getTileCount());
          llvm::errs() << " reduceK=";
          debugPrintValueOrConst(llvm::errs(), qa.getReduceK());
          llvm::errs() << "\n";
        }
        b.create<pynq::VectorInstrOp>(
            loc, i32c(static_cast<int32_t>(*bufIdOr)), qa.getTileCount(),
            qa.getReduceK(), i32c(static_cast<int32_t>(pynq::VectorOpCode::kQAdd)),
            i32c(static_cast<int32_t>(*extraIdOr)));
        toErase.push_back(&op);
        continue;
      }

      if (auto copy = llvm::dyn_cast<pynq::CopyOp>(&op)) {
        Value src = copy.getSrc();
        Value dst = copy.getDst();

        bool srcMem = src.getType().isa<MemRefType>();
        bool dstMem = dst.getType().isa<MemRefType>();
        bool srcBuf = isPynqBuffer(src);
        bool dstBuf = isPynqBuffer(dst);

        if (srcMem && dstBuf) {
          auto bufIdOr = getAllocatedBufferId(dst, &op);
          if (failed(bufIdOr)) {
            failedAny = true;
            return;
          }
          auto bytesOr = getStaticMemRefByteSize(src, &op, "copy src");
          auto tilesOr = computeTileCountFromLastDim(src, &op);
          if (failed(bytesOr) || failed(tilesOr)) {
            failedAny = true;
            return;
          }
          int32_t pkg = static_cast<int32_t>(
              (static_cast<int64_t>(*bytesOr) + DMAConfig::kBytesPerPackage - 1) /
              DMAConfig::kBytesPerPackage);

          if (debugOpConvert) {
            llvm::errs() << "[pynq-mid-lower][convert] CopyOp memref->buffer ";
            loc.print(llvm::errs());
            llvm::errs() << " dstBuffer=" << *bufIdOr
                         << " bytes=" << *bytesOr
                         << " tiles=" << *tilesOr
                         << " pkg=" << pkg
                         << " dir=" << DMAConfig::kDirectionLoad << "\n";
          }

          b.create<pynq::DataTransferInstrOp>(
              loc, src,
              i32c(static_cast<int32_t>(*bufIdOr)),
              i32c(*tilesOr),
              i32c(pkg),
              i32c(DMAConfig::kDirectionLoad));
          // Host->device transfer is asynchronous; ensure completion before
          // subsequent potentially-dependent ops.
          b.create<pynq::SyncOp>(loc);
          toErase.push_back(&op);
          continue;
        }

        if (srcBuf && dstMem) {
          auto bufIdOr = getAllocatedBufferId(src, &op);
          if (failed(bufIdOr)) {
            failedAny = true;
            return;
          }
          auto bytesOr = getStaticMemRefByteSize(dst, &op, "copy dst");
          auto tilesOr = computeTileCountFromLastDim(dst, &op);
          if (failed(bytesOr) || failed(tilesOr)) {
            failedAny = true;
            return;
          }
          int32_t pkg = static_cast<int32_t>(
              (static_cast<int64_t>(*bytesOr) + DMAConfig::kBytesPerPackage - 1) /
              DMAConfig::kBytesPerPackage);

          if (debugOpConvert) {
            llvm::errs() << "[pynq-mid-lower][convert] CopyOp buffer->memref ";
            loc.print(llvm::errs());
            llvm::errs() << " srcBuffer=" << *bufIdOr
                         << " bytes=" << *bytesOr
                         << " tiles=" << *tilesOr
                         << " pkg=" << pkg
                         << " dir=" << DMAConfig::kDirectionStore << "\n";
          }

          b.create<pynq::DataTransferInstrOp>(
              loc, dst,
              i32c(static_cast<int32_t>(*bufIdOr)),
              i32c(*tilesOr),
              i32c(pkg),
              i32c(DMAConfig::kDirectionStore));
          // Device->host transfer is asynchronous; sync before host consumes data.
          b.create<pynq::SyncOp>(loc);
          toErase.push_back(&op);
          continue;
        }

        if (srcMem && dstMem) {
          op.emitError() << "pynq.copy memref->memref not supported in PYNQMidLower";
          failedAny = true;
          return;
        }
        if (srcBuf && dstBuf) {
          op.emitError() << "pynq.copy buffer->buffer should be lowered earlier";
          failedAny = true;
          return;
        }
      }
    }

    for (Operation *op : toErase)
      op->erase();
  });

  if (failedAny)
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::allo::createPYNQMidLowerPass() {
  return std::make_unique<PYNQMidLowerPass>();
}
