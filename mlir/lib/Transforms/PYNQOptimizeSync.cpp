/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/PYNQOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"

#include <optional>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

namespace {

struct AccessSummary {
  llvm::SmallDenseSet<int32_t> reads;
  llvm::SmallDenseSet<int32_t> writes;
  // Bitmask of async instruction kinds seen since last kept sync.
  // If the pre-sync segment and post-sync segment share any kind,
  // we conservatively keep the sync (hardware engines may not auto-block).
  uint8_t kindMask = 0;
  bool unknown = false;
};

enum InstrKindBits : uint8_t {
  kMatMulKind = 1u << 0,
  kVectorKind = 1u << 1,
  kDataTransferKind = 1u << 2,
};

static std::optional<int64_t> tryGetConstI64(Value v) {
  if (!v)
    return std::nullopt;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
      return ia.getInt();
  }
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return std::nullopt;
}

static bool isIgnorableForSyncAnalysis(Operation *op) {
  return isa<arith::ConstantOp, arith::ConstantIntOp, memref::GetGlobalOp,
             memref::CastOp>(op);
}

static AccessSummary getAccessSummary(Operation *op) {
  AccessSummary a;

  if (auto mm = dyn_cast<pynq::MatMulInstrOp>(op)) {
    a.kindMask |= kMatMulKind;
    auto in = tryGetConstI64(mm.getInputBufferId());
    auto w = tryGetConstI64(mm.getWeightBufferId());
    auto out = tryGetConstI64(mm.getOutputBufferId());
    if (!in || !w || !out) {
      a.unknown = true;
      return a;
    }
    a.reads.insert(static_cast<int32_t>(*in));
    a.reads.insert(static_cast<int32_t>(*w));
    a.writes.insert(static_cast<int32_t>(*out));
    return a;
  }

  if (auto vec = dyn_cast<pynq::VectorInstrOp>(op)) {
    a.kindMask |= kVectorKind;
    auto buf = tryGetConstI64(vec.getBufferId());
    auto opCode = tryGetConstI64(vec.getOp());
    if (!buf || !opCode) {
      a.unknown = true;
      return a;
    }
    int32_t b = static_cast<int32_t>(*buf);
    a.reads.insert(b);
    a.writes.insert(b); // conservative: in-place vector op

    // IMPORTANT: extra_buffer_id is only semantically used by QAdd (op==1).
    // Other vector ops do not use extra_buffer_id, even if it is present in the
    // instruction encoding.
    if (*opCode == 1) {
      auto extra = tryGetConstI64(vec.getExtraBufferId());
      if (!extra) {
        a.unknown = true;
        return a;
      }
      // extra_buffer_id==0 is a valid on-chip buffer id; do NOT treat it as "unused".
      a.reads.insert(static_cast<int32_t>(*extra));
    }
    return a;
  }

  if (auto dt = dyn_cast<pynq::DataTransferInstrOp>(op)) {
    a.kindMask |= kDataTransferKind;
    auto dir = tryGetConstI64(dt.getDirection());
    auto buf = tryGetConstI64(dt.getBufferId());
    if (!dir || !buf) {
      a.unknown = true;
      return a;
    }
    int32_t b = static_cast<int32_t>(*buf);
    // 0 = host->device (write buffer), 1 = device->host (read buffer)
    // 2/3 are FIFO transfers (scale pack); treat as no buffer hazard.
    if (*dir == 0) {
      a.writes.insert(b);
      return a;
    }
    if (*dir == 1) {
      a.reads.insert(b);
      return a;
    }
    return a;
  }

  // Unknown op in the window: be conservative.
  a.unknown = true;
  return a;
}

static std::pair<uint8_t, bool>
collectNextSegmentKinds(Block::iterator startIt, Block &block) {
  uint8_t mask = 0;
  bool unknown = false;
  for (auto it = startIt; it != block.end(); ++it) {
    Operation *op = &*it;
    if (isIgnorableForSyncAnalysis(op))
      continue;
    if (isa<pynq::SyncOp>(op))
      break;
    if (isa<pynq::MatMulInstrOp, pynq::VectorInstrOp, pynq::DataTransferInstrOp>(op)) {
      auto acc = getAccessSummary(op);
      mask |= acc.kindMask;
      if (acc.unknown) {
        unknown = true;
        break;
      }
      continue;
    }
    // Unknown op in the next segment: be conservative.
    unknown = true;
    break;
  }
  return {mask, unknown};
}

static bool hasConflict(const AccessSummary &segment, const AccessSummary &next) {
  if (segment.unknown || next.unknown)
    return true;

  // RAW: next reads what segment wrote.
  for (int32_t r : next.reads)
    if (segment.writes.contains(r))
      return true;

  // WAR: next writes what segment read.
  for (int32_t w : next.writes)
    if (segment.reads.contains(w))
      return true;

  // WAW: next writes what segment wrote.
  for (int32_t w : next.writes)
    if (segment.writes.contains(w))
      return true;

  return false;
}

class PYNQOptimizeSyncPass
    : public PassWrapper<PYNQOptimizeSyncPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQOptimizeSyncPass)

  StringRef getArgument() const final { return "pynq-optimize-sync"; }
  StringRef getDescription() const final {
    return "Elide redundant pynq.sync ops using conservative buffer dependency checks";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;

      // Keep it simple & safe: only optimize single-block functions.
      if (!funcOp.getBody().hasOneBlock())
        return;

      Block &block = funcOp.getBody().front();

      AccessSummary sinceLastKeptSync;
      SmallVector<pynq::SyncOp> toErase;

      for (auto it = block.begin(); it != block.end(); ++it) {
        Operation *op = &*it;

        if (auto sync = dyn_cast<pynq::SyncOp>(op)) {
          // Find the next non-ignorable op (may be another sync).
          Operation *nextOp = nullptr;
          for (auto jt = std::next(it); jt != block.end(); ++jt) {
            Operation *cand = &*jt;
            if (isIgnorableForSyncAnalysis(cand))
              continue;
            nextOp = cand;
            break;
          }

          // Back-to-back sync: keep only the later one.
          if (nextOp && isa<pynq::SyncOp>(nextOp)) {
            toErase.push_back(sync);
            continue;
          }

          if (!nextOp) {
            // End of block: keep sync (conservative).
            sinceLastKeptSync = AccessSummary();
            continue;
          }

          if (isIgnorableForSyncAnalysis(nextOp)) {
            // Should be filtered above; keep conservative.
            sinceLastKeptSync = AccessSummary();
            continue;
          }

          AccessSummary nextAcc;
          if (isa<pynq::MatMulInstrOp, pynq::VectorInstrOp,
                 pynq::DataTransferInstrOp>(nextOp)) {
            nextAcc = getAccessSummary(nextOp);
          } else {
            // Unknown op after sync: do not elide.
            sinceLastKeptSync = AccessSummary();
            continue;
          }

          // New rule: if the segment before this sync and the segment after this
          // sync contain the same kind of async instruction, keep the sync.
          // (Different kinds do not require this rule; they only need buffer hazard checks.)
          auto nextKinds = collectNextSegmentKinds(std::next(it), block);
          uint8_t nextKindMask = nextKinds.first;
          bool nextSegUnknown = nextKinds.second;
          if (sinceLastKeptSync.unknown || nextSegUnknown) {
            // Not sure: keep sync.
            sinceLastKeptSync = AccessSummary();
            continue;
          }
          if ((sinceLastKeptSync.kindMask & nextKindMask) != 0) {
            sinceLastKeptSync = AccessSummary();
            continue;
          }

          if (!hasConflict(sinceLastKeptSync, nextAcc)) {
            // No dependency between (prevSync, sync) window and next op.
            // Defer waiting; remove this sync.
            toErase.push_back(sync);
            continue;
          }

          // Keep sync: it becomes a barrier.
          sinceLastKeptSync = AccessSummary();
          continue;
        }

        if (isIgnorableForSyncAnalysis(op))
          continue;

        // Accumulate accesses for known async hardware ops.
        if (isa<pynq::MatMulInstrOp, pynq::VectorInstrOp,
               pynq::DataTransferInstrOp>(op)) {
          AccessSummary cur = getAccessSummary(op);
          if (cur.unknown)
            sinceLastKeptSync.unknown = true;
          else {
            sinceLastKeptSync.reads.insert(cur.reads.begin(), cur.reads.end());
            sinceLastKeptSync.writes.insert(cur.writes.begin(), cur.writes.end());
            sinceLastKeptSync.kindMask |= cur.kindMask;
          }
          continue;
        }

        // Unknown op in window.
        sinceLastKeptSync.unknown = true;
      }

      for (auto s : toErase)
        s.erase();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::allo::createPYNQOptimizeSyncPass() {
  return std::make_unique<PYNQOptimizeSyncPass>();
}
