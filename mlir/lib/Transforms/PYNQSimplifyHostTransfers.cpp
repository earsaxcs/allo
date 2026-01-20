/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYNQ Simplify Host Transfers Pass (scaffolding)
 *
 * Motivation
 * ----------
 * After `lower-vivado-to-pynq`, the IR commonly contains a mix of:
 *   - host-side `memref<...>` values (RAM / host-visible memory)
 *   - on-chip `!pynq.buffer<...>` values (virtual or physical buffer)
 *   - `pynq.copy` / `pynq.data_transfer` endpoints bridging the two
 *
 * In many cases, intermediate values are copied buffer -> memref -> buffer
 * even though no host-only operation requires the round-trip. Those memrefs
 * are effectively temporary and can be eliminated.
 *
 * This pass is intended to:
 *   1) Analyze the compute/dataflow to identify host round-trips that are
 *      removable, using dominance and a simple notion of liveness.
 *   2) Rewrite the IR so data stays on buffers whenever possible, only
 *      materializing host memrefs when required by non-PYNQ ops.
 *   3) Enable better buffer reuse downstream (e.g., PYNQBufferAllocation).
 *
 * NOTE
 * ----
 * For this conversation turn, we only provide a *planning-oriented skeleton*.
 * The pass currently performs conservative pattern discovery and can emit
 * diagnostic remarks, but it intentionally does not rewrite the IR yet.
 */

#include "PassDetail.h"

#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pynq-simplify-host-transfers"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

namespace {

/// A discovered "round-trip" candidate:
///   %tmp = (host memref)
///   pynq.copy %buf -> %tmp
///   ... (no relevant users of %tmp besides the next copy)
///   pynq.copy %tmp -> %buf2
///
/// In the full implementation, this can often be simplified to:
///   pynq.copy %buf -> %buf2
/// and %tmp (and its defining ops like subview) can be DCE'd.
struct HostRoundTripCandidate {
  pynq::CopyOp bufToHost;
  pynq::CopyOp hostToBuf;
  Value hostValue; // memref-like SSA value bridging the two
};

struct HostSliceKey {
  Value base;
  Type elementType;

  // Physical offset (in elements) from `base`.
  int64_t offset = 0;

  // If true, this slice describes a contiguous physical range of
  // `contiguousElemCount` elements starting at `offset`.
  bool isContiguous = false;
  int64_t contiguousElemCount = 0;

  // Fallback: non-contiguous layout is described by (sizes, strides).
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<int64_t, 4> strides;
};

static bool operator==(const HostSliceKey &a, const HostSliceKey &b) {
  if (a.base != b.base)
    return false;
  if (a.elementType != b.elementType)
    return false;
  if (a.offset != b.offset)
    return false;
  if (a.isContiguous != b.isContiguous)
    return false;
  if (a.isContiguous)
    return a.contiguousElemCount == b.contiguousElemCount;
  return a.sizes == b.sizes && a.strides == b.strides;
}

static uint64_t hashHostSliceKey(const HostSliceKey &k) {
  auto rangeHash = [](auto &range) -> uint64_t {
    return static_cast<uint64_t>(llvm::hash_combine_range(range.begin(), range.end()));
  };
  uint64_t elemTyHash = static_cast<uint64_t>(llvm::hash_value(k.elementType.getAsOpaquePointer()));
  if (k.isContiguous) {
    return static_cast<uint64_t>(llvm::hash_combine(
        k.base.getAsOpaquePointer(), elemTyHash, k.offset, k.isContiguous,
        k.contiguousElemCount));
  }
  return static_cast<uint64_t>(llvm::hash_combine(
      k.base.getAsOpaquePointer(), elemTyHash, k.offset, k.isContiguous,
      rangeHash(k.sizes), rangeHash(k.strides)));
}

static bool isBufferUntouchedBetween(Value buf, Operation *from, Operation *to) {
  if (!buf || !from || !to)
    return false;
  if (from->getBlock() != to->getBlock())
    return false;

  Operation *cur = from->getNextNode();
  while (cur && cur != to) {
    for (Value operand : cur->getOperands()) {
      if (operand == buf)
        return false;
    }
    cur = cur->getNextNode();
  }
  return cur == to;
}

static bool isPynqBuffer(Value v) {
  return v && llvm::isa<pynq::BufferType>(v.getType());
}

static bool isHostMemref(Value v) {
  return v && v.getType().isa<MemRefType>();
}

static bool isViewLikeMemrefOp(Operation *op) {
  return llvm::isa_and_nonnull<memref::SubViewOp, memref::CastOp,
                               memref::CollapseShapeOp, memref::ExpandShapeOp,
                               memref::ReshapeOp, memref::ReinterpretCastOp>(op);
}

static Value getViewSource(Value v) {
  if (!v)
    return v;
  if (auto sub = v.getDefiningOp<memref::SubViewOp>())
    return sub.getSource();
  if (auto cast = v.getDefiningOp<memref::CastOp>())
    return cast.getSource();
  if (auto col = v.getDefiningOp<memref::CollapseShapeOp>())
    return col.getSrc();
  if (auto exp = v.getDefiningOp<memref::ExpandShapeOp>())
    return exp.getSrc();
  if (auto resh = v.getDefiningOp<memref::ReshapeOp>())
    return resh.getSource();
  if (auto ric = v.getDefiningOp<memref::ReinterpretCastOp>())
    return ric.getSource();
  return Value();
}

static Value getBaseMemref(Value v) {
  Value cur = v;
  while (cur && isHostMemref(cur)) {
    Value next = getViewSource(cur);
    if (!next)
      break;
    cur = next;
  }
  return cur;
}

static bool isLocalAllocMemref(Value v) {
  if (!v || !v.getType().isa<MemRefType>())
    return false;
  if (v.isa<BlockArgument>())
    return false;
  Operation *def = v.getDefiningOp();
  return llvm::isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(def);
}

static bool matchStaticIndexOfr(OpFoldResult ofr, int64_t &out) {
  // NOTE: Use OpFoldResult's API. Mixing it with llvm::dyn_cast on unrelated
  // wrapper types can lead to UB.
  if (Attribute attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = llvm::dyn_cast<IntegerAttr>(attr)) {
      out = ia.getInt();
      return true;
    }
    return false;
  }

  Value v = ofr.dyn_cast<Value>();
  if (!v)
    return false;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = llvm::dyn_cast<IntegerAttr>(cst.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  return false;
}

static bool isStaticContiguous(ArrayRef<int64_t> shape,
                               ArrayRef<int64_t> strides) {
  if (shape.size() != strides.size())
    return false;
  if (shape.empty())
    return true;
  if (strides.back() != 1)
    return false;

  int64_t running = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] == ShapedType::kDynamic)
      return false;
    if (strides[i] == ShapedType::kDynamic)
      return false;
    if (i == static_cast<int64_t>(shape.size()) - 1) {
      if (strides[i] != 1)
        return false;
      running = shape[i];
      continue;
    }
    if (strides[i] != running)
      return false;
    // Prevent overflow; if it would overflow, treat as non-contiguous.
    if (shape[i] != 0 && running > (std::numeric_limits<int64_t>::max() / shape[i]))
      return false;
    running *= shape[i];
  }
  return true;
}

static std::optional<int64_t> staticNumElements(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic)
      return std::nullopt;
    if (d != 0 && n > (std::numeric_limits<int64_t>::max() / d))
      return std::nullopt;
    n *= d;
  }
  return n;
}

static std::optional<HostSliceKey> getHostSliceKey(Value v) {
  if (!v || !v.getType().isa<MemRefType>())
    return std::nullopt;

  HostSliceKey key;
  key.base = getBaseMemref(v);
  if (!key.base || !key.base.getType().isa<MemRefType>())
    return std::nullopt;

  auto memrefTy = v.getType().cast<MemRefType>();
  key.elementType = memrefTy.getElementType();

  // Require static shapes; this pass is intentionally conservative.
  if (!memrefTy.hasStaticShape())
    return std::nullopt;

  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(memrefTy, strides, offset)))
    return std::nullopt;
  if (offset == ShapedType::kDynamic)
    return std::nullopt;
  for (int64_t s : strides) {
    if (s == ShapedType::kDynamic)
      return std::nullopt;
  }
  key.offset = offset;

  ArrayRef<int64_t> shape = memrefTy.getShape();

  // Prefer a rank-agnostic contiguous-range key when possible.
  if (isStaticContiguous(shape, strides)) {
    auto nElems = staticNumElements(shape);
    if (!nElems)
      return std::nullopt;
    key.isContiguous = true;
    key.contiguousElemCount = *nElems;
    return key;
  }

  key.isContiguous = false;
  key.sizes.assign(shape.begin(), shape.end());
  key.strides.assign(strides.begin(), strides.end());
  return key;
}

static bool baseMemrefOnlyUsedByCopiesAndViews(Value base) {
  if (!base || !base.getType().isa<MemRefType>())
    return false;

  llvm::SmallVector<Value, 32> worklist;
  llvm::SmallPtrSet<void *, 32> visited;
  worklist.push_back(base);
  visited.insert(base.getAsOpaquePointer());

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      if (llvm::isa<pynq::CopyOp>(user))
        continue;
      if (isViewLikeMemrefOp(user)) {
        // Continue exploring through view results.
        for (Value res : user->getResults()) {
          if (!res.getType().isa<MemRefType>())
            continue;
          void *p = res.getAsOpaquePointer();
          if (visited.insert(p).second)
            worklist.push_back(res);
        }
        continue;
      }
      return false;
    }
  }
  return true;
}

/// Return true if `v` is a trivially "view-like" op over another memref.
///
/// We keep this as scaffolding. In the full implementation we will:
///   - prove the view is identity (offset==0, same underlying buffer)
///   - or model subview ranges precisely (requires offset support)
static bool isTrivialMemrefView(Value /*v*/) {
  return false;
}

/// In a future implementation, this will peel away view ops (cast/reshape/subview)
/// when they don't affect physical layout, so we can reason about true storage.
static Value canonicalizeStorageValue(Value v) {
  // TODO: implement view peeling / identity checks.
  return v;
}

static bool hasOnlyCopyUsers(Value hostValue,
                            pynq::CopyOp bufToHost,
                            pynq::CopyOp hostToBuf) {
  // A conservative check: require exact two uses (the two copy ops).
  // TODO: relax to allow view ops + canonicalize.
  unsigned uses = 0;
  for (OpOperand &use : hostValue.getUses()) {
    (void)use;
    ++uses;
    if (uses > 2)
      return false;
  }
  if (uses != 2)
    return false;

  bool seenIn = false;
  bool seenOut = false;
  for (Operation *user : hostValue.getUsers()) {
    if (user == bufToHost.getOperation())
      seenIn = true;
    if (user == hostToBuf.getOperation())
      seenOut = true;
  }
  return seenIn && seenOut;
}

class PYNQSimplifyHostTransfersPass
    : public PassWrapper<PYNQSimplifyHostTransfersPass, OperationPass<ModuleOp>> {
public:
  PYNQSimplifyHostTransfersPass() = default;
  PYNQSimplifyHostTransfersPass(const PYNQSimplifyHostTransfersPass &pass)
      : PassWrapper(pass) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQSimplifyHostTransfersPass)

  StringRef getArgument() const final { return "pynq-simplify-host-transfers"; }

  StringRef getDescription() const final {
    return "Analyze and (eventually) remove redundant host memref round-trips between PYNQ buffers";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  // Options: kept for future expansion.
  Option<bool> emitRemarks{*this, "emit-remarks",
                           llvm::cl::desc("Emit remarks for detected simplification candidates"),
                           llvm::cl::init(false)};

  Option<bool> dryRun{*this, "dry-run",
                      llvm::cl::desc("Do not rewrite IR; only analyze and report"),
                      llvm::cl::init(true)};

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk functions; the transformation will be function-local initially.
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;
      analyzeFunction(funcOp);
    }
  }

private:
  void analyzeFunction(func::FuncOp funcOp) {
    DominanceInfo dom(funcOp);
    Liveness liveness(funcOp);
    (void)dom;
    (void)liveness;

    // Collect copy ops in program order.
    llvm::SmallVector<pynq::CopyOp, 64> copies;
    funcOp.walk([&](pynq::CopyOp copy) { copies.push_back(copy); });

    // Fast position index for "latest dominating write" selection.
    llvm::DenseMap<Operation *, int32_t> opPosition;
    opPosition.reserve(copies.size());
    for (int32_t i = 0, e = static_cast<int32_t>(copies.size()); i < e; ++i)
      opPosition[copies[i].getOperation()] = i;

    // Group copies by local alloc memref bases. We only touch memrefs that are:
    //  - local alloc/alloca
    //  - only used via view ops and pynq.copy
    llvm::DenseMap<Value, llvm::SmallVector<pynq::CopyOp, 16>> copiesByBase;
    for (pynq::CopyOp copy : copies) {
      Value src = copy.getSrc();
      Value dst = copy.getDst();
      if (isHostMemref(src)) {
        Value base = getBaseMemref(src);
        if (isLocalAllocMemref(base))
          copiesByBase[base].push_back(copy);
      }
      if (isHostMemref(dst)) {
        Value base = getBaseMemref(dst);
        if (isLocalAllocMemref(base))
          copiesByBase[base].push_back(copy);
      }
    }

    IRRewriter rewriter(funcOp.getContext());
    bool changed = false;
    size_t rewrittenReads = 0;

    for (auto &it : copiesByBase) {
      Value base = it.first;
      auto &baseCopies = it.second;
      if (!baseMemrefOnlyUsedByCopiesAndViews(base))
        continue;

      struct WriteRec {
        pynq::CopyOp op;
        HostSliceKey key;
        uint64_t hash;
        int32_t pos;
      };
      struct ReadRec {
        pynq::CopyOp op;
        HostSliceKey key;
        uint64_t hash;
        int32_t pos;
      };

      llvm::DenseMap<uint64_t, llvm::SmallVector<WriteRec, 8>> writes;
      llvm::SmallVector<ReadRec, 32> reads;

      // Classify copies that touch this base memref.
      for (pynq::CopyOp copy : baseCopies) {
        Value src = copy.getSrc();
        Value dst = copy.getDst();
        const int32_t pos = opPosition.lookup(copy.getOperation());

        if (isPynqBuffer(src) && isHostMemref(dst) && getBaseMemref(dst) == base) {
          auto keyOpt = getHostSliceKey(dst);
          if (!keyOpt)
            continue;
          HostSliceKey key = *keyOpt;
          uint64_t h = hashHostSliceKey(key);
          writes[h].push_back(WriteRec{copy, key, h, pos});
          continue;
        }

        if (isHostMemref(src) && isPynqBuffer(dst) && getBaseMemref(src) == base) {
          auto keyOpt = getHostSliceKey(src);
          if (!keyOpt)
            continue;
          HostSliceKey key = *keyOpt;
          uint64_t h = hashHostSliceKey(key);
          reads.push_back(ReadRec{copy, key, h, pos});
          continue;
        }
      }

      if (reads.empty() || writes.empty())
        continue;

      struct PlannedRewrite {
        pynq::CopyOp read;
        Value srcBuf;
        Value dstBuf;
      };

      llvm::SmallVector<PlannedRewrite, 32> planned;
      llvm::SmallPtrSet<Operation *, 32> opsToErase;

      // For each read (host->buf), try to bypass via the latest dominating write (buf->host).
      for (auto &read : reads) {
        auto wIt = writes.find(read.hash);
        if (wIt == writes.end())
          continue;

        WriteRec *best = nullptr;
        for (WriteRec &w : wIt->second) {
          if (!(w.key == read.key))
            continue;
          if (!dom.dominates(w.op.getOperation(), read.op.getOperation()))
            continue;
          if (!best || w.pos > best->pos)
            best = &w;
        }
        if (!best)
          continue;

        Value srcBuf = best->op.getSrc();
        Value dstBuf = read.op.getDst();
        if (!isPynqBuffer(srcBuf) || !isPynqBuffer(dstBuf))
          continue;

        // Safety checks:
        // - Require same buffer type (capacity/element type).
        // - Require the source buffer is not used between the write and the read.
        // This keeps the rewrite conservative until we implement a proper side-effect model.
        if (srcBuf.getType() != dstBuf.getType())
          continue;
        if (!isBufferUntouchedBetween(srcBuf, best->op.getOperation(), read.op.getOperation()))
          continue;

        // NOTE: We intentionally do NOT elide same-buffer readback yet;
        // it may be semantically meaningful if the buffer was modified.
        if (srcBuf == dstBuf)
          continue;

        if (emitRemarks) {
          read.op.emitRemark("rewrite host->buf to buf->buf (bypass local host memref)");
          best->op.emitRemark("paired dominating buf->host write used for bypass");
        }
        if (!dryRun)
          planned.push_back(PlannedRewrite{read.op, srcBuf, dstBuf});
      }

      // Apply rewrites after analysis decisions are made.
      // This avoids mutating the IR while still consulting dominance info.
      if (!dryRun) {
        for (PlannedRewrite &p : planned) {
          if (!p.read || !p.read.getOperation())
            continue;
          rewriter.setInsertionPoint(p.read);
          rewriter.create<pynq::CopyOp>(p.read.getLoc(), p.srcBuf, p.dstBuf);
          opsToErase.insert(p.read.getOperation());
          ++rewrittenReads;
          changed = true;
        }
      }

      if (!dryRun) {
        // If we eliminated all reads from this local base, then the host staging is unobservable.
        // We can drop all remaining copies that touch this base.
        bool allReadsEliminated = true;
        for (auto &read : reads) {
          if (!opsToErase.contains(read.op.getOperation())) {
            allReadsEliminated = false;
            break;
          }
        }
        if (allReadsEliminated) {
          for (auto &pair : writes) {
            for (WriteRec &w : pair.second)
              opsToErase.insert(w.op.getOperation());
          }
        }

        // Erase marked copy ops.
        // (We do this before view cleanup so use-lists settle.)
        for (pynq::CopyOp copy : baseCopies) {
          if (opsToErase.contains(copy.getOperation()))
            copy.erase();
        }

        // Cleanup dead view/alloc ops rooted at base. Keep it local and simple.
        bool erasedSomething = true;
        bool baseDefErased = false;
        while (erasedSomething) {
          erasedSomething = false;
          for (Operation *user : llvm::make_early_inc_range(base.getUsers())) {
            if (isViewLikeMemrefOp(user) && user->use_empty()) {
              user->erase();
              erasedSomething = true;
            }
          }
          if (auto def = base.getDefiningOp()) {
            if (llvm::isa<memref::AllocOp, memref::AllocaOp>(def) && def->use_empty()) {
              def->erase();
              erasedSomething = true;
              baseDefErased = true;
            }
          }

          // IMPORTANT: once the defining op of `base` is erased, `base` becomes
          // a dangling Value handle. Do not touch `base.getUsers()` again.
          if (baseDefErased)
            break;
        }
      }
    }

    if (emitRemarks) {
      funcOp.emitRemark() << "pynq-simplify-host-transfers: rewritten "
                          << rewrittenReads
                          << " host->buf reads (dry-run=" << (dryRun ? "true" : "false") << ")";
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[pynq-simplify-host-transfers] " << funcOp.getName() << ": "
                   << rewrittenReads << " reads rewritten";
      if (dryRun)
        llvm::dbgs() << " (dry-run)";
      llvm::dbgs() << "\n";
    });

    (void)changed;
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createPYNQSimplifyHostTransfersPass() {
  return std::make_unique<PYNQSimplifyHostTransfersPass>();
}

} // namespace allo
} // namespace mlir
