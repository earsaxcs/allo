/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// PYNQSplitForwardInlineCleanup Pass
//
// Target split-forward pattern introduced by allo-split-forward-batch-loop:
//   wrapper scf.for { %r = call @inner(...); memref.copy %r, %outSubview }
//
// This pass inlines eligible inner bodies at wrapper call sites, remaps
// returned alloc values to wrapper out subviews, and removes redundant
// call-result copy chains.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/PYNQOps.h"
#include "allo/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace allo;

namespace {

static constexpr StringLiteral kInnerSplitAttr = "allo.batch.split_forward";
static constexpr StringLiteral kOuterSplitAttr =
    "allo.batch.split_forward.wrapper";

static bool isStaticRowMajorContiguous(MemRefType type,
                                       bool requireZeroOffset) {
  if (!type || !type.hasStaticShape())
    return false;

  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return false;
  if (requireZeroOffset) {
    if (offset == ShapedType::kDynamic || offset != 0)
      return false;
  }
  if (strides.size() != type.getRank())
    return false;
  for (int64_t stride : strides) {
    if (stride == ShapedType::kDynamic)
      return false;
  }

  int64_t expected = 1;
  auto shape = type.getShape();
  for (int64_t i = type.getRank(); i > 0; --i) {
    int64_t idx = i - 1;
    if (strides[idx] != expected)
      return false;
    expected *= shape[idx];
  }

  return true;
}

struct InnerRewriteState {
  func::FuncOp func;
  func::ReturnOp returnOp;
  SmallVector<Value> returnedValues;
  SmallVector<memref::AllocOp> returnedAllocs;
  SmallVector<MemRefType> resultTypes;
};

LogicalResult analyzeInnerFunction(func::FuncOp func, InnerRewriteState &state,
                                   std::string &reason) {
  if (!func->hasAttr(kInnerSplitAttr)) {
    reason = "missing split-forward inner attribute";
    return failure();
  }
  if (func.isDeclaration()) {
    reason = "function is declaration";
    return failure();
  }
  if (!func.getBody().hasOneBlock()) {
    reason = "function does not have a single block body";
    return failure();
  }

  FunctionType fnType = func.getFunctionType();
  if (fnType.getNumResults() == 0) {
    reason = "function has no results";
    return failure();
  }

  SmallVector<MemRefType> resultTypes;
  resultTypes.reserve(fnType.getNumResults());
  for (Type t : fnType.getResults()) {
    auto mt = dyn_cast<MemRefType>(t);
    if (!mt) {
      reason = "function has non-memref result";
      return failure();
    }
    resultTypes.push_back(mt);
  }

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  if (returns.size() != 1) {
    reason = "function does not have exactly one return";
    return failure();
  }

  func::ReturnOp ret = returns.front();
  if (ret.getNumOperands() != static_cast<unsigned>(resultTypes.size())) {
    reason = "return operand count does not match function results";
    return failure();
  }

  SmallVector<Value> returnedValues;
  SmallVector<memref::AllocOp> returnedAllocs;
  returnedValues.reserve(resultTypes.size());
  returnedAllocs.reserve(resultTypes.size());

  for (auto [idx, operand] : llvm::enumerate(ret.getOperands())) {
    if (operand.getType() != resultTypes[idx]) {
      reason = "return operand type does not match function result type";
      return failure();
    }
    auto alloc = operand.getDefiningOp<memref::AllocOp>();
    if (!alloc) {
      reason = "returned value is not directly produced by memref.alloc";
      return failure();
    }
    returnedValues.push_back(operand);
    returnedAllocs.push_back(alloc);
  }

  state = InnerRewriteState{func, ret, returnedValues, returnedAllocs,
                            resultTypes};
  return success();
}

bool inlineWrapperCall(func::CallOp call, InnerRewriteState &state,
                       std::string &reason) {
  auto parentFunc = call->getParentOfType<func::FuncOp>();
  if (!parentFunc || !parentFunc->hasAttr(kOuterSplitAttr)) {
    reason = "call is not inside split-forward wrapper";
    return false;
  }

  if (call.getNumResults() != state.resultTypes.size()) {
    reason = "call result count does not match inner results";
    return false;
  }

  SmallVector<Value> outOperands;
  outOperands.reserve(state.resultTypes.size());
  SmallVector<memref::CopyOp> copiesToErase;
  copiesToErase.reserve(state.resultTypes.size());
  SmallVector<memref::CastOp> sourceCastsToErase;
  sourceCastsToErase.reserve(state.resultTypes.size());

  for (auto [idx, callResult] : llvm::enumerate(call.getResults())) {
    if (!callResult.hasOneUse()) {
      reason = "call result is not single-use";
      return false;
    }

    memref::CopyOp copy;
    auto directUser = *callResult.user_begin();
    if (auto directCopy = dyn_cast<memref::CopyOp>(directUser)) {
      if (directCopy.getSource() != callResult) {
        reason = "memref.copy source is not call result";
        return false;
      }
      copy = directCopy;
    } else if (auto sourceCast = dyn_cast<memref::CastOp>(directUser)) {
      if (!sourceCast.getResult().hasOneUse()) {
        reason = "source cast from call result is not single-use";
        return false;
      }
      auto castUser = *sourceCast.getResult().user_begin();
      auto castCopy = dyn_cast<memref::CopyOp>(castUser);
      if (!castCopy || castCopy.getSource() != sourceCast.getResult()) {
        reason = "call result cast user is not memref.copy source";
        return false;
      }
      copy = castCopy;
      sourceCastsToErase.push_back(sourceCast);
    } else {
      reason = "call result user is neither memref.copy nor memref.cast->memref.copy";
      return false;
    }

    Value outOperand = copy.getTarget();
    auto outTy = dyn_cast<MemRefType>(outOperand.getType());
    if (!outTy) {
      reason = "copy target is not memref";
      return false;
    }
    if (outTy != state.resultTypes[idx]) {
      if (!memref::CastOp::areCastCompatible(outTy, state.resultTypes[idx])) {
        reason = "copy target type is not cast-compatible with call result";
        return false;
      }
      if (!isStaticRowMajorContiguous(outTy, /*requireZeroOffset=*/false)) {
        reason = "contiguous_cast requires statically row-major input";
        return false;
      }
      if (!isStaticRowMajorContiguous(state.resultTypes[idx],
                                      /*requireZeroOffset=*/true)) {
        reason = "contiguous_cast requires zero-offset row-major output";
        return false;
      }
      OpBuilder castBuilder(copy);
      outOperand = castBuilder
                       .create<pynq::ContiguousCastOp>(call.getLoc(),
                                                       state.resultTypes[idx],
                                                       outOperand)
                       .getOutput();
    }

    outOperands.push_back(outOperand);
    copiesToErase.push_back(copy);
  }

  if (copiesToErase.empty()) {
    reason = "no memref.copy users found for call results";
    return false;
  }

  // Insert inline body right before the last copy so all outOperands
  // (including subviews/casts) are already defined and dominate their uses.
  Operation *insertBefore = copiesToErase.back().getOperation();
  OpBuilder inlineBuilder(insertBefore);

  IRMapping mapper;
  for (auto [arg, actual] : llvm::zip(state.func.getArguments(), call.getOperands()))
    mapper.map(arg, actual);
  for (auto [ret, out] : llvm::zip(state.returnedValues, outOperands))
    mapper.map(ret, out);

  llvm::SmallPtrSet<Operation *, 4> returnedAllocOps;
  for (memref::AllocOp alloc : state.returnedAllocs)
    returnedAllocOps.insert(alloc.getOperation());

  for (Operation &opRef : state.func.front()) {
    Operation *op = &opRef;
    if (op == state.returnOp)
      continue;
    if (returnedAllocOps.contains(op))
      continue;
    inlineBuilder.clone(*op, mapper);
  }

  for (memref::CopyOp copy : copiesToErase)
    copy.erase();
  for (memref::CastOp cast : sourceCastsToErase)
    cast.erase();

  call.erase();
  return true;
}

static bool eliminateWrapperInputCopies(func::FuncOp wrapper) {
  // Currently no use, because it tests offset as well
  // auto isStaticRowMajorContiguous = [](MemRefType type) {
  //   SmallVector<int64_t, 4> strides;
  //   int64_t offset = 0;
  //   if (failed(getStridesAndOffset(type, strides, offset)))
  //     return false;
  //   if (offset != 0)
  //     return false;
  //   if (strides.size() != type.getRank())
  //     return false;
  //   int64_t expected = 1;
  //   for (int64_t i = type.getRank() - 1; i >= 0; --i) {
  //     if (strides[i] != expected)
  //       return false;
  //     expected *= type.getShape()[i];
  //   }
  //   return true;
  // };

  SmallVector<memref::CopyOp> copies;
  wrapper.walk([&](memref::CopyOp copy) { copies.push_back(copy); });

  bool changed = false;
  for (memref::CopyOp copy : copies) {
    if (!copy)
      continue;

    Value src = copy.getSource();
    Value dst = copy.getTarget();
    auto dstAlloc = dst.getDefiningOp<memref::AllocOp>();
    auto srcSubview = src.getDefiningOp<memref::SubViewOp>();
    if (!dstAlloc || !srcSubview)
      continue;

    auto srcTy = dyn_cast<MemRefType>(src.getType());
    auto dstTy = dyn_cast<MemRefType>(dst.getType());
    if (!srcTy || !dstTy)
      continue;
    // if (!isStaticRowMajorContiguous(srcTy))
    //   continue;
    if (!memref::CastOp::areCastCompatible(srcTy, dstTy))
      continue;

    SmallVector<OpOperand *> readUses;
    for (OpOperand &use : dst.getUses()) {
      if (use.getOwner() == copy)
        continue;

      auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
      auto transposeOp = dyn_cast<pynq::ActivationLayoutTransposeOp>(use.getOwner());
      if ((!linalgOp || !linalgOp.isDpsInput(&use)) && (!transposeOp || transposeOp.getInputMutable() != use)) {
        readUses.clear();
        break;
      }
      readUses.push_back(&use);
    }

    if (readUses.empty())
      continue;

    if (!isStaticRowMajorContiguous(srcTy, /*requireZeroOffset=*/false))
      continue;
    if (!isStaticRowMajorContiguous(dstTy, /*requireZeroOffset=*/true))
      continue;

    OpBuilder b(copy);
    Value castedSrc =
        b.create<pynq::ContiguousCastOp>(copy.getLoc(), dstTy, src)
            .getOutput();

    for (OpOperand *use : readUses)
      use->set(castedSrc);

    copy.erase();
    if (dst.use_empty())
      dstAlloc.erase();
    changed = true;
  }

  return changed;
}

} // namespace

namespace mlir {
namespace allo {

bool applyPYNQSplitForwardInlineCleanup(ModuleOp &module) {
  SmallVector<func::FuncOp> innerFuncs;
  module.walk([&](func::FuncOp func) {
    if (func->hasAttr(kInnerSplitAttr))
      innerFuncs.push_back(func);
  });

  bool hadError = false;
  for (func::FuncOp inner : innerFuncs) {
    InnerRewriteState state;
    std::string reason;
    if (failed(analyzeInnerFunction(inner, state, reason))) {
      inner.emitRemark() << "[pynq-split-forward-inline-cleanup] skipped: "
                         << reason;
      continue;
    }

    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == inner.getSymName())
        calls.push_back(call);
    });

    for (func::CallOp call : calls) {
      std::string callReason;
      if (!inlineWrapperCall(call, state, callReason)) {
        call.emitRemark()
            << "[pynq-split-forward-inline-cleanup] skip call inlining: "
            << callReason;
      }
    }

    if (inner.use_empty())
      inner.erase();
  }

  module.walk([&](func::FuncOp func) {
    if (func->hasAttr(kOuterSplitAttr))
      (void)eliminateWrapperInputCopies(func);
  });

  return !hadError;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloPYNQSplitForwardInlineCleanup
    : public PYNQSplitForwardInlineCleanupBase<
          AlloPYNQSplitForwardInlineCleanup> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!applyPYNQSplitForwardInlineCleanup(module))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>>
createPYNQSplitForwardInlineCleanupPass() {
  return std::make_unique<AlloPYNQSplitForwardInlineCleanup>();
}

} // namespace allo
} // namespace mlir
