/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYNQ Elide Redundant Copies Pass
 *
 * This pass runs after higher-level copy simplifications (e.g.
 * PYNQSimplifyHostTransfers) and tries to further reduce buffer usage by
 * forwarding reads of a PYNQ buffer to the buffer that most recently defined
 * its contents via pynq.copy.
 *
 * Conceptually, this is a conservative, local "copy forwarding" / "copy
 * propagation" for !pynq.buffer values:
 *
 *   pynq.copy %a, %b
 *   use(%b)
 *   pynq.copy %c, %b
 *   use(%b)
 *
 * can become:
 *   use(%a)
 *   use(%c)
 *
 * and the now-dead copies to %b can be erased.
 *
 * Correctness & Scope
 * -------------------
 * - We are conservative around side effects: unknown ops that touch PYNQ buffers
 *   are treated as read+write barriers.
 * - For now, the implementation is block-local (single basic block). This
 *   avoids subtle CFG path/merge issues. A follow-up can extend to CFG
 *   dataflow using MLIR's DataFlowFramework.
 * - Snapshot semantics: forwarding %b -> %a is only correct if %a's contents
 *   are unchanged between the defining copy and the use. Locally we approximate
 *   this with a per-buffer "version" incremented on writes.
 */

#include "PassDetail.h"

#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pynq-elide-redundant-copies"

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

/// A small, extensible effect model for PYNQ buffers.
///
/// We intentionally do not rely on MemoryEffectOpInterface here because most
/// PYNQ ops are currently not modeled with memory effects.
struct BufferEffectModel {
  BufferAccessKind getAccess(Operation *op, Value buf) const {
    if (!op || !buf)
      return BufferAccessKind::Unknown;

    // Known ops with explicit buffer semantics.
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

    // Vector ops.
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
        return BufferAccessKind::ReadWrite; // in-place add
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

    // For other pynq ops, be conservative: any buffer operand is treated as
    // read+write unless we know better.
    if (op->getDialect() && op->getDialect()->getNamespace() == "pynq") {
      for (Value operand : op->getOperands()) {
        if (operand == buf)
          return BufferAccessKind::ReadWrite;
      }
      return BufferAccessKind::None;
    }

    // Non-pynq ops: if they mention a pynq buffer at all, treat as unknown.
    for (Value operand : op->getOperands()) {
      if (operand == buf)
        return BufferAccessKind::Unknown;
    }
    return BufferAccessKind::None;
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

struct ContentInfo {
  uint64_t token = 0;
};

static bool isBlockArgument(Value v) {
  return v && llvm::isa<BlockArgument>(v);
}

static bool hasAnyUseAfterInBlock(Value v, Operation *anchor,
                                 Operation *exclude) {
  if (!v || !anchor)
    return true;
  Block *block = anchor->getBlock();
  if (!block)
    return true;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user || user == exclude)
      continue;
    if (user->getBlock() != block)
      return true; // outside-block use: be conservative
    if (anchor->isBeforeInBlock(user))
      return true;
  }
  return false;
}

static bool hasAnyReadUseAfterInBlock(Value v, Operation *anchor,
                                     Operation *exclude,
                                     const BufferEffectModel &effects) {
  if (!v || !anchor)
    return true;
  Block *block = anchor->getBlock();
  if (!block)
    return true;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user || user == exclude)
      continue;
    if (user->getBlock() != block)
      return true; // outside-block use: be conservative
    if (!anchor->isBeforeInBlock(user))
      continue;
    if (effects.isUnknown(user, v))
      return true;
    if (effects.isRead(user, v))
      return true;
  }
  return false;
}

static Operation *getLastUserAfterInBlock(Value v, Operation *anchor,
                                         Operation *exclude) {
  if (!v || !anchor)
    return nullptr;
  Block *block = anchor->getBlock();
  if (!block)
    return nullptr;
  Operation *last = nullptr;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user || user == exclude)
      continue;
    if (user->getBlock() != block)
      return nullptr; // outside-block use: refuse to reason
    if (!anchor->isBeforeInBlock(user))
      continue;
    if (!last || last->isBeforeInBlock(user))
      last = user;
  }
  return last;
}

static bool hasAnyUseInOpenClosedRangeInBlock(Value v, Operation *after,
                                             Operation *beforeInclusive,
                                             Operation *exclude,
                                             const BufferEffectModel &effects) {
  if (!v || !after || !beforeInclusive)
    return true;
  Block *block = after->getBlock();
  if (!block || beforeInclusive->getBlock() != block)
    return true;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user || user == exclude)
      continue;
    if (user->getBlock() != block)
      return true;
    if (!after->isBeforeInBlock(user))
      continue;
    // user is after `after`; now require user <= beforeInclusive
    if (beforeInclusive->isBeforeInBlock(user))
      continue;
    // Any interaction counts as interference.
    if (effects.isUnknown(user, v) || effects.isRead(user, v) ||
        effects.isWrite(user, v))
      return true;
  }
  return false;
}

static bool allUsesInSameBlock(Value v, Block *block) {
  if (!v || !block)
    return false;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user || user->getBlock() != block)
      return false;
  }
  return true;
}

static bool bufferAllocRoleIs(Value buf, StringRef role) {
  auto alloc = buf.getDefiningOp<pynq::BufferAllocOp>();
  if (!alloc)
    return false;
  auto roleAttr = alloc->getAttrOfType<StringAttr>("role");
  return roleAttr && roleAttr.getValue() == role;
}

/// Return true if `buf` has any read uses in the current IR.
static bool bufferHasReadUses(Value buf, const BufferEffectModel &effects) {
  if (!isPynqBuffer(buf))
    return false;
  for (OpOperand &use : buf.getUses()) {
    Operation *user = use.getOwner();
    if (!user)
      continue;
    if (effects.isRead(user, buf))
      return true;
  }
  return false;
}

/// Assign `buf` a new token, and keep canonical holder maps coherent.
static void assignNewToken(Value buf, uint64_t newToken,
                           llvm::DenseMap<Value, uint64_t> &bufferToken,
                           llvm::DenseMap<uint64_t, Value> &canonicalHolder) {
  if (!isPynqBuffer(buf))
    return;
  uint64_t old = bufferToken.lookup(buf);
  if (old != 0) {
    auto it = canonicalHolder.find(old);
    if (it != canonicalHolder.end() && it->second == buf)
      canonicalHolder.erase(it);
  }
  bufferToken[buf] = newToken;
  if (canonicalHolder.find(newToken) == canonicalHolder.end())
    canonicalHolder[newToken] = buf;
}

class PYNQElideRedundantCopiesPass
    : public PassWrapper<PYNQElideRedundantCopiesPass, OperationPass<ModuleOp>> {
public:
  PYNQElideRedundantCopiesPass() = default;
  PYNQElideRedundantCopiesPass(const PYNQElideRedundantCopiesPass &pass)
      : PassWrapper(pass) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQElideRedundantCopiesPass)

  StringRef getArgument() const final { return "pynq-elide-redundant-copies"; }

  StringRef getDescription() const final {
    return "Elide redundant pynq.copy by forwarding buffer reads";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
  }

  Option<bool> emitRemarks{
      *this, "emit-remarks",
      llvm::cl::desc("Emit remarks for copy-forwarding decisions"),
      llvm::cl::init(false)};

  Option<bool> dryRun{*this, "dry-run",
                      llvm::cl::desc("Do not rewrite IR; only analyze and report"),
                      llvm::cl::init(true)};

  Option<bool> blockLocalOnly{
      *this, "block-local-only",
      llvm::cl::desc(
          "Only perform forwarding within a single basic block (conservative)"),
      llvm::cl::init(true)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    BufferEffectModel effects;

    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;
      analyzeAndRewriteFunction(funcOp, effects);
    }
  }

private:
  void analyzeAndRewriteFunction(func::FuncOp funcOp,
                                 const BufferEffectModel &effects) {
    IRRewriter rewriter(funcOp.getContext());

    size_t forwardedOperands = 0;
    size_t erasedCopies = 0;
    size_t coalescedCopies = 0;
    size_t erasedAllocs = 0;

    for (Block &block : funcOp.getBody().getBlocks()) {
      if (!blockLocalOnly) {
        // TODO: CFG dataflow version (AbstractDenseDataFlowAnalysis).
        // Keep the initial implementation block-local.
      }

      // Snapshot semantics model:
      // Each write event produces a fresh token; buffer->buffer copy copies the
      // token from src to dst (snapshot). Forwarding is then based on tokens:
      // we can replace a read of %b with any canonical holder that currently
      // has the same token as %b.
      uint64_t nextToken = 1;
      llvm::DenseMap<Value, uint64_t> bufferToken;
      llvm::DenseMap<uint64_t, Value> canonicalHolder;

      int64_t pos = 0;
      for (Operation &op : llvm::make_early_inc_range(block.getOperations())) {
        ++pos;

        // Phase 0: destructive buffer reuse.
        //
        // Pattern: pynq.copy %src, %dst ; (uses of %dst ...)
        // If %src is not used after the copy (excluding the copy itself), we
        // can replace all later uses of %dst with %src and erase the copy.
        // This effectively reuses %src as the writable buffer and can remove
        // both the copy and the temporary allocation of %dst.
        if (auto copy = llvm::dyn_cast<pynq::CopyOp>(&op)) {
          Value src = copy.getSrc();
          Value dst = copy.getDst();
          if (isPynqBuffer(src) && isPynqBuffer(dst) && src != dst) {
            Block *b = copy->getBlock();
            // Be conservative: only reason about single-block uses.
            if (b && allUsesInSameBlock(src, b) && allUsesInSameBlock(dst, b)) {
              // Never write into external/unknown buffers.
              bool srcIsSafeForWrites = !isBlockArgument(src) &&
                                       !bufferAllocRoleIs(src, "weight") &&
                                       !bufferAllocRoleIs(src, "input") &&
                                       !bufferAllocRoleIs(src, "bias");

              // Safe destructive reuse condition:
              // We will rewrite future uses of %dst to %src. This is only
              // semantics-preserving if %src is not otherwise accessed (read
              // or written) during %dst's live range after the copy.
              Operation *lastDstUser = getLastUserAfterInBlock(dst, copy, copy);
              bool srcInterferes = true;
              if (lastDstUser) {
                srcInterferes = hasAnyUseInOpenClosedRangeInBlock(
                    src, copy, lastDstUser, copy, effects);
              }

              if (srcIsSafeForWrites && lastDstUser && !srcInterferes) {
                if (emitRemarks)
                  copy->emitRemark("coalesce copy by reusing source buffer")
                      << " src=" << src << " dst=" << dst;
                if (!dryRun) {
                  for (OpOperand &use :
                       llvm::make_early_inc_range(dst.getUses())) {
                    Operation *user = use.getOwner();
                    if (!user || user == copy)
                      continue;
                    if (!copy->isBeforeInBlock(user))
                      continue;
                    use.set(src);
                  }

                  // Erase the copy itself.
                  copy.erase();
                  ++coalescedCopies;

                  // If dst is now dead and it was a fresh alloc, erase it.
                  if (dst.use_empty()) {
                    if (auto alloc = dst.getDefiningOp<pynq::BufferAllocOp>()) {
                      alloc.erase();
                      ++erasedAllocs;
                    }
                  }
                }
                continue;
              }
            }
          }
        }

        auto ensureInitialized = [&](Value buf) {
          if (!isPynqBuffer(buf))
            return;
          if (bufferToken.lookup(buf) == 0)
            assignNewToken(buf, nextToken++, bufferToken, canonicalHolder);
        };

        auto clobberBuffer = [&](Value buf) {
          if (!isPynqBuffer(buf))
            return;
          assignNewToken(buf, nextToken++, bufferToken, canonicalHolder);
        };

        // Step 1: forward operands for reads.
        for (unsigned i = 0, e = op.getNumOperands(); i < e; ++i) {
          Value operand = op.getOperand(i);
          if (!isPynqBuffer(operand))
            continue;

          ensureInitialized(operand);

          // Per requirement: only focus on buffer<->buffer copies. We keep
          // memref<->buffer transfers untouched (even if forwarding would be
          // semantically valid) to avoid interacting with host transfer logic.
          if (auto copy = llvm::dyn_cast<pynq::CopyOp>(&op)) {
            Value other = (copy.getSrc() == operand) ? copy.getDst() : copy.getSrc();
            if (!(isPynqBuffer(operand) && isPynqBuffer(other))) {
              // Still need to model writes/unknowns for token correctness.
              if (effects.isUnknown(&op, operand))
                clobberBuffer(operand);
              else if (effects.isWrite(&op, operand))
                clobberBuffer(operand);
              continue;
            }
          }

          // Unknown non-pynq op touching a buffer: treat as barrier.
          if (effects.isUnknown(&op, operand)) {
            clobberBuffer(operand);
            continue;
          }

          // Only forward pure reads. For in-place/readwrite ops, rewriting the
          // operand would change the write target and is not semantics-preserving.
          if (effects.isRead(&op, operand) && !effects.isWrite(&op, operand)) {
            uint64_t tok = bufferToken.lookup(operand);
            Value holder = canonicalHolder.lookup(tok);
            if (!holder || holder == operand)
              continue;
            if (holder.getType() != operand.getType())
              continue;
            // Ensure the canonical holder still holds the same token.
            if (bufferToken.lookup(holder) != tok)
              continue;

            if (emitRemarks)
              op.emitRemark("forward buffer read")
                  << " operand " << i << " from " << operand << " to " << holder;
            if (!dryRun) {
              op.setOperand(i, holder);
              ++forwardedOperands;
            }
          }

          if (effects.isWrite(&op, operand))
            clobberBuffer(operand);
        }

        // Step 2: update token mapping for buffer<->buffer copies.
        if (auto copy = llvm::dyn_cast<pynq::CopyOp>(&op)) {
          Value src = copy.getSrc();
          Value dst = copy.getDst();
          if (isPynqBuffer(src) && isPynqBuffer(dst)) {
            ensureInitialized(src);
            ensureInitialized(dst);

            // dst now holds the same content token as src (snapshot).
            uint64_t tok = bufferToken.lookup(src);
            assignNewToken(dst, tok, bufferToken, canonicalHolder);
          }

          // Copies involving a buffer and memref are currently ignored here.
          continue;
        }
      }
    }

    // Cleanup: after forwarding, erase buffer<->buffer copies whose destination
    // buffer is never read in the (updated) IR. This is conservative and
    // directly matches the "a->b; use b" => "use a" use-case.
    if (!dryRun) {
      llvm::SmallVector<pynq::CopyOp, 32> candidates;
      funcOp.walk([&](pynq::CopyOp copy) {
        if (isPynqBuffer(copy.getSrc()) && isPynqBuffer(copy.getDst()))
          candidates.push_back(copy);
      });
      for (pynq::CopyOp copy : candidates) {
        Value dst = copy.getDst();
        if (!bufferHasReadUses(dst, effects)) {
          copy.erase();
          ++erasedCopies;
        }
      }
    }

    if (emitRemarks) {
      funcOp.emitRemark() << "pynq-elide-redundant-copies: forwarded "
                          << forwardedOperands << " operands, erased "
                          << erasedCopies << " copies, coalesced "
                          << coalescedCopies << " copies, erased "
                          << erasedAllocs << " allocs (dry-run="
                          << (dryRun ? "true" : "false") << ")";
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[pynq-elide-redundant-copies] " << funcOp.getName()
                   << ": forwarded=" << forwardedOperands
                   << ", erased=" << erasedCopies
                   << ", coalesced=" << coalescedCopies
                   << ", erasedAllocs=" << erasedAllocs;
      if (dryRun)
        llvm::dbgs() << " (dry-run)";
      llvm::dbgs() << "\n";
    });

    (void)rewriter;
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createPYNQElideRedundantCopiesPass() {
  return std::make_unique<PYNQElideRedundantCopiesPass>();
}

} // namespace allo
} // namespace mlir
