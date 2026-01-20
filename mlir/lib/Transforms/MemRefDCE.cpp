/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// MemRefDCE Pass
// This pass removes memrefs that are not loaded from.
// We only look at memrefs allocated in functions.
// Global memrefs and memrefs in function args are not removed.
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "allo/Transforms/Passes.h"

#include "allo/Dialect/PYNQOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void cleanUpUnusedOps(func::FuncOp &func) {
  SmallVector<Operation *, 32> toErase;
  func.walk([&](Operation *op) {
    if (op->getNumResults() != 0 && op->use_empty())
      toErase.push_back(op);
  });
  for (Operation *op : llvm::reverse(toErase))
    op->erase();
}

static bool isViewLikeMemrefOp(Operation *op) {
  return llvm::isa_and_nonnull<memref::SubViewOp, memref::CastOp,
                               memref::CollapseShapeOp,
                               memref::ExpandShapeOp>(op);
}

/// Returns true if `op` reads from `memrefValue`.
///
/// This pass is intentionally conservative: unknown ops are treated as reads
/// to avoid deleting allocations that might be observed.
static bool isReadFromMemref(Operation *op, Value memrefValue) {
  if (!op)
    return true;

  if (llvm::isa<memref::LoadOp, affine::AffineLoadOp>(op))
    return true;
  if (llvm::isa<func::ReturnOp, func::CallOp>(op))
    return true;

  if (auto copy = dyn_cast<mlir::allo::pynq::CopyOp>(op)) {
    // memref -> buffer is a read from memref.
    return copy.getSrc() == memrefValue;
  }

  if (auto memcpy = dyn_cast<memref::CopyOp>(op)) {
    return memcpy.getSource() == memrefValue;
  }

  // Stores are writes, not reads.
  if (llvm::isa<memref::StoreOp, affine::AffineStoreOp>(op))
    return false;

  // View ops don't read data; they just create aliases.
  if (isViewLikeMemrefOp(op))
    return false;

  // Unknown user: treat as read to be safe.
  return true;
}

static bool canEraseUserWhenNoReads(Operation *op, Value memrefValue) {
  if (!op)
    return false;
  if (isViewLikeMemrefOp(op))
    return true;
  if (auto copy = dyn_cast<mlir::allo::pynq::CopyOp>(op)) {
    // Only erase if this memref is the destination (i.e., write-only path).
    return copy.getDst() == memrefValue;
  }
  if (auto memcpy = dyn_cast<memref::CopyOp>(op)) {
    return memcpy.getTarget() == memrefValue;
  }
  if (llvm::isa<memref::StoreOp, affine::AffineStoreOp>(op))
    return true;
  return false;
}

/// Explore the alias/view graph starting from `root` (alloc result) and decide
/// whether there exists any read. If no reads, collect erase candidates.
static void analyzeMemrefUses(Value root,
                              bool &hasRead,
                              SmallVectorImpl<Operation *> &eraseCandidates) {
  hasRead = false;
  eraseCandidates.clear();

  SmallVector<Value, 16> worklist;
  llvm::SmallPtrSet<void *, 32> visitedValues;
  llvm::SmallPtrSet<Operation *, 32> visitedOps;

  worklist.push_back(root);
  visitedValues.insert(root.getAsOpaquePointer());

  while (!worklist.empty() && !hasRead) {
    Value cur = worklist.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      if (!visitedOps.insert(user).second)
        continue;

      if (isReadFromMemref(user, cur)) {
        hasRead = true;
        break;
      }

      // If we are in the no-read world, only erase users we know are safe.
      if (!canEraseUserWhenNoReads(user, cur)) {
        hasRead = true;
        break;
      }
      eraseCandidates.push_back(user);

      if (isViewLikeMemrefOp(user)) {
        for (Value res : user->getResults()) {
          if (!res.getType().isa<MemRefType>())
            continue;
          void *p = res.getAsOpaquePointer();
          if (visitedValues.insert(p).second)
            worklist.push_back(res);
        }
      }
    }
  }
}

static void eraseInUseSafeOrder(ArrayRef<Operation *> ops) {
  // Erase leaf-to-root: repeatedly erase ops whose results are unused.
  // This avoids aborting when erasing view ops that still feed other ops.
  SmallVector<Operation *, 64> work(ops.begin(), ops.end());
  bool progress = true;
  while (progress) {
    progress = false;
    for (Operation *&op : work) {
      if (!op)
        continue;
      if (op->getNumResults() != 0 && !op->use_empty())
        continue;
      op->erase();
      op = nullptr;
      progress = true;
    }
  }
}

void removeNeverLoadedMemRef(func::FuncOp &func) {
  SmallVector<Operation *, 8> memRefAllocOps;
  func.walk([&](Operation *op) {
    if (auto memRefAllocOp = dyn_cast<memref::AllocOp>(op)) {
      memRefAllocOps.push_back(memRefAllocOp);
    }
  });
  std::reverse(memRefAllocOps.begin(), memRefAllocOps.end());
  for (auto op : memRefAllocOps) {
    auto v = op->getResult(0);

    bool hasRead = false;
    SmallVector<Operation *, 64> eraseCandidates;
    analyzeMemrefUses(v, hasRead, eraseCandidates);
    if (hasRead)
      continue;

    // No reads: it is safe to erase the whole write-only subgraph + alloc.
    eraseCandidates.push_back(op);
    eraseInUseSafeOrder(eraseCandidates);
  }
}

/// Pass entry point
bool applyMemRefDCE(ModuleOp &mod) {
  for (auto func : mod.getOps<func::FuncOp>()) {
    removeNeverLoadedMemRef(func);
    cleanUpUnusedOps(func);
  }
  return true;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloMemRefDCETransformation
    : public MemRefDCEBase<AlloMemRefDCETransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyMemRefDCE(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createMemRefDCEPass() {
  return std::make_unique<AlloMemRefDCETransformation>();
}
} // namespace allo
} // namespace mlir