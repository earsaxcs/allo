/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// VivadoMergeRedundantQuantDequantPass
//
// Deduplicate boundary vivado.quant/vivado.dequant ops when they:
// - Share the same source value
// - Share the exact same quant parameters (scale/zero + ALL attributes)
// - Produce the same output type
//
// This is a conservative pass intended to run before ToggleVivadoTranspose.
// It only merges when it can prove that the produced buffers are not written
// after the quant/dequant op (via MemoryEffectOpInterface).
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/VivadoOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "vivado-merge-redundant-quant-dequant"

using namespace mlir;
using namespace mlir::allo;

namespace vivado_ops = mlir::allo::vivado;

namespace {

static bool hasUnknownOrWriteOrFreeEffect(Operation *user, Value v) {
  auto iface = dyn_cast<MemoryEffectOpInterface>(user);
  if (!iface)
    return true; // Conservative: unknown effects => do not merge.

  SmallVector<MemoryEffects::EffectInstance, 4> effects;
  iface.getEffectsOnValue(v, effects);
  for (auto &e : effects) {
    if (isa<MemoryEffects::Write>(e.getEffect()) ||
        isa<MemoryEffects::Free>(e.getEffect()))
      return true;
  }
  return false;
}

static bool isBufferWrittenElsewhere(Value buffer, Operation *allowedWriter) {
  for (OpOperand &use : buffer.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == allowedWriter)
      continue;
    if (hasUnknownOrWriteOrFreeEffect(owner, buffer))
      return true;
  }
  return false;
}

template <typename QuantLikeOp>
static bool opsMatchIgnoringOutput(QuantLikeOp a, QuantLikeOp b) {
  if (!a || !b)
    return false;
  if (a->getNumOperands() != b->getNumOperands())
    return false;

  // Output types must match. (We will redirect the output buffer.)
  if (a->getOperand(0).getType() != b->getOperand(0).getType())
    return false;

  // All non-output operands must be exactly identical SSA values.
  for (unsigned i = 1; i < a->getNumOperands(); ++i) {
    if (a->getOperand(i) != b->getOperand(i))
      return false;
  }

  // Attribute dictionary must be exactly identical.
  if (a->getAttrDictionary() != b->getAttrDictionary())
    return false;

  return true;
}

template <typename QuantLikeOp>
static bool tryMergeInto(QuantLikeOp current, QuantLikeOp canonical,
                         DominanceInfo &dom) {
  if (!opsMatchIgnoringOutput(current, canonical))
    return false;

  // Must dominate: canonical must execute before current and before all uses we
  // rewrite.
  if (!dom.dominates(canonical.getOperation(), current.getOperation()))
    return false;

  Value canonicalOut = canonical->getOperand(0);
  Value currentOut = current->getOperand(0);

  // Be conservative: do not merge if either output buffer is written elsewhere.
  if (isBufferWrittenElsewhere(canonicalOut, canonical.getOperation()))
    return false;
  if (isBufferWrittenElsewhere(currentOut, current.getOperation()))
    return false;

  // Ensure every rewritten use is read-only (or otherwise known-safe) and is
  // dominated by canonical.
  for (OpOperand &use : llvm::make_early_inc_range(currentOut.getUses())) {
    Operation *user = use.getOwner();
    if (user == current.getOperation())
      continue;

    // Don't touch unknown/write/free users.
    if (hasUnknownOrWriteOrFreeEffect(user, currentOut))
      return false;

    if (!dom.dominates(canonical.getOperation(), user))
      return false;
  }

  // Rewrite uses (excluding the op being erased).
  for (OpOperand &use : llvm::make_early_inc_range(currentOut.getUses())) {
    Operation *user = use.getOwner();
    if (user == current.getOperation())
      continue;
    use.set(canonicalOut);
  }

  // Erase the redundant quant/dequant.
  Operation *toErase = current.getOperation();
  toErase->erase();

  // Best-effort cleanup: erase now-dead alloc for the redundant output.
  if (Operation *def = currentOut.getDefiningOp()) {
    if (def->use_empty()) {
      if (isa<memref::AllocOp, memref::AllocaOp>(def)) {
        def->erase();
      }
    }
  }

  return true;
}

static int64_t mergeInFunc(func::FuncOp func) {
  DominanceInfo dom(func);
  int64_t mergedCount = 0;

  // We process per-block in textual order. This catches the common case where
  // boundary ops are inserted at multiple use sites in the same block.
  for (Block &block : func.getBody().getBlocks()) {
    llvm::SmallVector<vivado_ops::QuantOp, 8> quantCanonicals;
    llvm::SmallVector<vivado_ops::DequantOp, 8> dequantCanonicals;

    for (auto it = block.begin(), e = block.end(); it != e;) {
      Operation *op = &*it++;

      if (auto q = dyn_cast<vivado_ops::QuantOp>(op)) {
        bool merged = false;
        for (auto canonical : quantCanonicals) {
          if (tryMergeInto(q, canonical, dom)) {
            ++mergedCount;
            merged = true;
            break;
          }
        }
        if (!merged)
          quantCanonicals.push_back(q);
        continue;
      }

      if (auto dq = dyn_cast<vivado_ops::DequantOp>(op)) {
        bool merged = false;
        for (auto canonical : dequantCanonicals) {
          if (tryMergeInto(dq, canonical, dom)) {
            ++mergedCount;
            merged = true;
            break;
          }
        }
        if (!merged)
          dequantCanonicals.push_back(dq);
        continue;
      }
    }
  }

  return mergedCount;
}

struct VivadoMergeRedundantQuantDequantPass
    : public PassWrapper<VivadoMergeRedundantQuantDequantPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VivadoMergeRedundantQuantDequantPass)

  StringRef getArgument() const override {
    return "vivado-merge-redundant-quant-dequant";
  }
  StringRef getDescription() const override {
    return "Merge identical vivado.quant/vivado.dequant with same source and parameters";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vivado_ops::VivadoDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    if (!applyVivadoMergeRedundantQuantDequant(module, context))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

bool applyVivadoMergeRedundantQuantDequant(ModuleOp &module,
                                          MLIRContext *context) {
  (void)context;
  int64_t totalMerged = 0;

  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;
    totalMerged += mergeInFunc(func);
  });

  // Always succeed; this is an optimization pass.
  (void)totalMerged;
  return true;
}

std::unique_ptr<OperationPass<ModuleOp>>
createVivadoMergeRedundantQuantDequantPass() {
  return std::make_unique<VivadoMergeRedundantQuantDequantPass>();
}

} // namespace allo
} // namespace mlir
