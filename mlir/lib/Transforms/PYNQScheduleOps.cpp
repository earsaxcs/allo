/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYNQ Block-local Scheduling Pass
 *
 * Goal: conservatively reorder a subset of pynq ops within a single basic block
 * to reduce the peak number of simultaneously-live PYNQ buffers.
 *
 * Correctness-first constraints:
 * - Only schedules within a contiguous "window" of known PYNQ ops.
 * - Does not move ops across barriers (unknown ops, pynq.sync, terminators).
 * - Preserves SSA def-use dependencies.
 * - Preserves per-buffer ordering constraints for writes (and read-after-write).
 *
 * Default mode is dry-run (analyze + remarks only).
 */

#include "PassDetail.h"

#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pynq-schedule-ops"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

namespace {

static bool isPynqBuffer(Value v) {
  return v && llvm::isa<pynq::BufferType>(v.getType());
}

enum class BufferAccessKind : uint8_t {
  None,
  Read,
  Write,
  ReadWrite,
  Unknown,
};

struct BufferEffectModel {
  BufferAccessKind getAccess(Operation *op, Value buf) const {
    if (!op || !buf)
      return BufferAccessKind::Unknown;

    if (auto copy = llvm::dyn_cast<pynq::CopyOp>(op)) {
      bool r = (copy.getSrc() == buf);
      bool w = (copy.getDst() == buf);
      if (r && w)
        return BufferAccessKind::ReadWrite;
      if (r)
        return BufferAccessKind::Read;
      if (w)
        return BufferAccessKind::Write;
      return BufferAccessKind::None;
    }

    if (auto mm = llvm::dyn_cast<pynq::MatMulOp>(op)) {
      if (mm.getInputBuffer() == buf || mm.getWeightBuffer() == buf)
        return BufferAccessKind::Read;
      if (mm.getOutputBuffer() == buf)
        return BufferAccessKind::Write;
      return BufferAccessKind::None;
    }

    // Vector ops (in-place unless explicitly modeled otherwise).
    if (auto gelu = llvm::dyn_cast<pynq::GELUOp>(op)) {
      if (gelu.getBuffer() == buf)
        return BufferAccessKind::ReadWrite;
      return BufferAccessKind::None;
    }
    if (auto sm = llvm::dyn_cast<pynq::SoftmaxOp>(op)) {
      if (sm.getBuffer() == buf)
        return BufferAccessKind::ReadWrite;
      return BufferAccessKind::None;
    }
    if (auto ln = llvm::dyn_cast<pynq::LayerNormOp>(op)) {
      if (ln.getBuffer() == buf)
        return BufferAccessKind::ReadWrite;
      return BufferAccessKind::None;
    }
    if (auto qa = llvm::dyn_cast<pynq::QAddOp>(op)) {
      if (qa.getBuffer() == buf)
        return BufferAccessKind::ReadWrite;
      if (qa.getExtraBuffer() == buf)
        return BufferAccessKind::Read;
      return BufferAccessKind::None;
    }
    if (auto vo = llvm::dyn_cast<pynq::VectorOp>(op)) {
      if (vo.getBuffer() == buf)
        return BufferAccessKind::ReadWrite;
      if (vo.getExtraBuffer() == buf)
        return BufferAccessKind::Read;
      return BufferAccessKind::None;
    }

    // Unknown ops are treated as barriers by the scheduler.
    return BufferAccessKind::Unknown;
  }

  bool isRead(Operation *op, Value buf) const {
    auto k = getAccess(op, buf);
    return k == BufferAccessKind::Read || k == BufferAccessKind::ReadWrite;
  }

  bool isWrite(Operation *op, Value buf) const {
    auto k = getAccess(op, buf);
    return k == BufferAccessKind::Write || k == BufferAccessKind::ReadWrite;
  }

  bool isUnknown(Operation *op, Value buf) const {
    return getAccess(op, buf) == BufferAccessKind::Unknown;
  }
};

static bool hasUseAfterInBlock(Value v, Operation *anchor) {
  if (!v || !anchor)
    return true;
  Block *block = anchor->getBlock();
  if (!block)
    return true;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user)
      continue;
    if (user->getBlock() != block)
      return true; // outside-block: conservative
    if (anchor->isBeforeInBlock(user))
      return true;
  }
  return false;
}

static bool isBufferToBufferCopy(pynq::CopyOp copy) {
  return isPynqBuffer(copy.getSrc()) && isPynqBuffer(copy.getDst());
}

static bool isSchedulablePynqOp(Operation *op) {
  if (!op)
    return false;
  if (llvm::isa<pynq::SyncOp>(op))
    return false;

  // Only schedule a conservative subset of ops.
  if (auto copy = llvm::dyn_cast<pynq::CopyOp>(op))
    return isBufferToBufferCopy(copy);
  if (llvm::isa<pynq::MatMulOp, pynq::GELUOp, pynq::SoftmaxOp,
                pynq::LayerNormOp, pynq::QAddOp, pynq::VectorOp>(op))
    return true;

  return false;
}

static bool isBarrier(Operation *op) {
  if (!op)
    return true;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return true;
  if (llvm::isa<pynq::SyncOp>(op))
    return true;
  return !isSchedulablePynqOp(op);
}

struct WindowScheduleStats {
  unsigned peakBefore = 0;
  unsigned peakAfter = 0;
  bool changed = false;
};

static unsigned computePeakLiveBuffers(ArrayRef<Operation *> order,
                                      const llvm::DenseMap<Value, int> &initUses,
                                      const llvm::DenseMap<Value, bool> &useAfter,
                                      const llvm::SmallVectorImpl<Value> &buffers) {
  llvm::DenseMap<Value, int> remaining = initUses;

  auto liveCount = [&]() -> unsigned {
    unsigned c = 0;
    for (Value b : buffers) {
      int r = remaining.lookup(b);
      bool after = useAfter.lookup(b);
      if (r > 0 || after)
        ++c;
    }
    return c;
  };

  unsigned peak = liveCount();
  for (Operation *op : order) {
    for (Value operand : op->getOperands()) {
      if (!isPynqBuffer(operand))
        continue;
      int r = remaining.lookup(operand);
      remaining[operand] = r - 1;
    }
    peak = std::max(peak, liveCount());
  }
  return peak;
}

static WindowScheduleStats scheduleWindow(Block &block,
                                         ArrayRef<Operation *> windowOps,
                                         const BufferEffectModel &effects,
                                         bool dryRun, bool onlyIfImproves,
                                         bool emitRemarks) {
  WindowScheduleStats stats;
  if (windowOps.size() < 2)
    return stats;

  llvm::DenseSet<Operation *> windowSet;
  windowSet.reserve(windowOps.size());
  for (Operation *op : windowOps)
    windowSet.insert(op);

  Operation *lastOp = windowOps.back();

  // Collect buffers touched in this window.
  llvm::SmallVector<Value, 32> buffers;
  {
    llvm::SmallPtrSet<Value, 32> seen;
    for (Operation *op : windowOps) {
      for (Value operand : op->getOperands()) {
        if (!isPynqBuffer(operand))
          continue;
        if (seen.insert(operand).second)
          buffers.push_back(operand);
      }
    }
  }

  // For each buffer, count uses inside window and whether it has any use after
  // the window (outside-block or later ops).
  llvm::DenseMap<Value, int> initUses;
  llvm::DenseMap<Value, bool> hasUseAfter;
  initUses.reserve(buffers.size());
  hasUseAfter.reserve(buffers.size());

  for (Value buf : buffers) {
    int count = 0;
    for (Operation *op : windowOps) {
      for (Value operand : op->getOperands()) {
        if (operand == buf)
          ++count;
      }
    }
    initUses[buf] = count;
    hasUseAfter[buf] = hasUseAfterInBlock(buf, lastOp);
  }

  // Build dependency graph within window.
  llvm::DenseMap<Operation *, unsigned> indegree;
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 8>> succs;
  llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 8>> succSet;
  llvm::DenseMap<Operation *, unsigned> originalPos;

  indegree.reserve(windowOps.size());
  succs.reserve(windowOps.size());
  succSet.reserve(windowOps.size());
  originalPos.reserve(windowOps.size());

  for (unsigned i = 0; i < windowOps.size(); ++i) {
    Operation *op = windowOps[i];
    indegree[op] = 0;
    originalPos[op] = i;
  }

  auto addEdge = [&](Operation *pred, Operation *succ) {
    if (!pred || !succ || pred == succ)
      return;
    auto &set = succSet[pred];
    if (!set.insert(succ).second)
      return;
    succs[pred].push_back(succ);
    indegree[succ] = indegree.lookup(succ) + 1;
  };

  // SSA def-use edges.
  for (Operation *op : windowOps) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && windowSet.contains(def))
        addEdge(def, op);
    }
  }

  // Per-buffer ordering edges (preserve write semantics).
  llvm::DenseMap<Value, Operation *> lastAccess;
  llvm::DenseMap<Value, Operation *> lastWriter;
  for (Operation *op : windowOps) {
    for (Value operand : op->getOperands()) {
      if (!isPynqBuffer(operand))
        continue;

      if (effects.isUnknown(op, operand)) {
        // Should not happen: unknown ops are treated as barriers.
        continue;
      }

      bool r = effects.isRead(op, operand);
      bool w = effects.isWrite(op, operand);

      if (w) {
        if (Operation *la = lastAccess.lookup(operand))
          addEdge(la, op);
      } else if (r) {
        if (Operation *lw = lastWriter.lookup(operand))
          addEdge(lw, op);
      }

      if (r || w) {
        lastAccess[operand] = op;
        if (w)
          lastWriter[operand] = op;
      }
    }
  }

  // Initialize ready set.
  llvm::SmallVector<Operation *, 32> ready;
  ready.reserve(windowOps.size());
  for (Operation *op : windowOps)
    if (indegree.lookup(op) == 0)
      ready.push_back(op);

  llvm::DenseMap<Value, int> remainingUses = initUses;
  llvm::SmallVector<Operation *, 64> newOrder;
  newOrder.reserve(windowOps.size());

  auto computeScore = [&](Operation *op) -> std::pair<int64_t, unsigned> {
    llvm::SmallPtrSet<Value, 8> uniq;
    unsigned freed = 0;
    for (Value operand : op->getOperands()) {
      if (!isPynqBuffer(operand))
        continue;
      if (!uniq.insert(operand).second)
        continue;
      if (remainingUses.lookup(operand) == 1 && !hasUseAfter.lookup(operand))
        ++freed;
    }
    // Higher is better.
    int64_t score = static_cast<int64_t>(freed) * 1024 -
                    static_cast<int64_t>(uniq.size());
    return {score, originalPos.lookup(op)};
  };

  while (!ready.empty()) {
    // Pick best ready op.
    unsigned bestIdx = 0;
    auto bestKey = computeScore(ready[0]);
    for (unsigned i = 1; i < ready.size(); ++i) {
      auto key = computeScore(ready[i]);
      if (key.first > bestKey.first ||
          (key.first == bestKey.first && key.second < bestKey.second)) {
        bestIdx = i;
        bestKey = key;
      }
    }

    Operation *op = ready[bestIdx];
    ready.erase(ready.begin() + bestIdx);
    newOrder.push_back(op);

    // Update remaining uses for buffers.
    llvm::SmallPtrSet<Value, 8> uniq;
    for (Value operand : op->getOperands()) {
      if (!isPynqBuffer(operand))
        continue;
      // Decrement once per operand occurrence (consistent with initUses).
      int r = remainingUses.lookup(operand);
      remainingUses[operand] = r - 1;
    }

    // Release successors.
    for (Operation *succ : succs.lookup(op)) {
      unsigned &deg = indegree[succ];
      if (deg == 0)
        continue;
      --deg;
      if (deg == 0)
        ready.push_back(succ);
    }
  }

  if (newOrder.size() != windowOps.size()) {
    // Cyclic deps (shouldn't happen). Bail out.
    return stats;
  }

  stats.peakBefore = computePeakLiveBuffers(windowOps, initUses, hasUseAfter,
                                            buffers);
  stats.peakAfter = computePeakLiveBuffers(newOrder, initUses, hasUseAfter,
                                           buffers);

  bool orderDiffers = false;
  for (unsigned i = 0; i < windowOps.size(); ++i) {
    if (windowOps[i] != newOrder[i]) {
      orderDiffers = true;
      break;
    }
  }

  bool improves = stats.peakAfter < stats.peakBefore;
  bool allowed = orderDiffers && (!onlyIfImproves || improves);

  if (emitRemarks) {
    // Emit on the last op so remarks show up near the region of interest.
    lastOp->emitRemark() << "pynq-schedule-ops window: peak "
                         << stats.peakBefore << " -> " << stats.peakAfter
                         << ", rewritten="
                         << ((allowed && !dryRun) ? "true" : "false")
                         << ", dry-run=" << (dryRun ? "true" : "false");
  }

  if (!dryRun && allowed) {
    // Rewrite: move ops to match newOrder, preserving relative placement of
    // non-window ops and barriers.
    Block::iterator insertIt = windowOps.front()->getIterator();
    for (Operation *op : newOrder) {
      op->moveBefore(&block, insertIt);
      insertIt = std::next(op->getIterator());
    }
    stats.changed = true;
  }

  return stats;
}

class PYNQScheduleOpsPass
    : public PassWrapper<PYNQScheduleOpsPass, OperationPass<ModuleOp>> {
public:
  PYNQScheduleOpsPass() = default;
  PYNQScheduleOpsPass(const PYNQScheduleOpsPass &pass) : PassWrapper(pass) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQScheduleOpsPass)

  StringRef getArgument() const final { return "pynq-schedule-ops"; }
  StringRef getDescription() const final {
    return "Conservatively schedule pynq ops to reduce peak live buffers";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
  }

  Option<bool> emitRemarks{
      *this, "emit-remarks",
      llvm::cl::desc("Emit remarks for scheduling decisions"),
      llvm::cl::init(false)};

  Option<bool> dryRun{
      *this, "dry-run",
      llvm::cl::desc("Do not rewrite IR; only analyze (default)"),
      llvm::cl::init(true)};

  Option<bool> onlyIfImproves{
      *this, "only-if-improves",
      llvm::cl::desc("Only rewrite when estimated peak decreases"),
      llvm::cl::init(true)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    BufferEffectModel effects;

    unsigned numWindows = 0;
    unsigned numRewritten = 0;

    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;

      for (Block &block : funcOp.getBody().getBlocks()) {
        llvm::SmallVector<Operation *, 64> window;

        for (Operation &op : block.getOperations()) {
          if (isBarrier(&op)) {
            if (!window.empty()) {
              ++numWindows;
              auto stats = scheduleWindow(block, window, effects, dryRun,
                                          onlyIfImproves, emitRemarks);
              if (stats.changed)
                ++numRewritten;
              window.clear();
            }
            continue;
          }

          window.push_back(&op);
        }

        if (!window.empty()) {
          ++numWindows;
          auto stats = scheduleWindow(block, window, effects, dryRun,
                                      onlyIfImproves, emitRemarks);
          if (stats.changed)
            ++numRewritten;
          window.clear();
        }
      }
    }

    if (emitRemarks) {
      module.emitRemark() << "pynq-schedule-ops: windows=" << numWindows
                          << ", rewritten=" << numRewritten
                          << " (dry-run=" << (dryRun ? "true" : "false")
                          << ")";
    }
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createPYNQScheduleOpsPass() {
  return std::make_unique<PYNQScheduleOpsPass>();
}

} // namespace allo
} // namespace mlir
