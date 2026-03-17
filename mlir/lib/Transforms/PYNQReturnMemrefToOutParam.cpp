/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// PYNQReturnMemrefToOutParam Pass
//
// Rewrite strict single-memref-return functions to explicit out-parameter form:
//   func @foo(...) -> memref<...>
// becomes:
//   func @foo(..., %out: memref<...>)
//
// and rewrites matched call sites from result-using calls to void calls with a
// freshly-created memref.alloc output operand.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Transforms/Passes.h"

#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace allo;

namespace {

struct FunctionRewriteState {
  func::FuncOp func;
  func::ReturnOp returnOp;
  memref::AllocOp returnedAlloc;
  Value returnedValue;
  MemRefType resultType;
};

LogicalResult analyzeFunctionForRewrite(func::FuncOp func,
                                        FunctionRewriteState &state,
                                        std::string &reason) {
  if (func.isDeclaration()) {
    reason = "function is declaration (no body)";
    return failure();
  }

  FunctionType fnType = func.getFunctionType();
  if (fnType.getNumResults() != 1) {
    reason = "function result count is not exactly 1";
    return failure();
  }

  auto resultType = dyn_cast<MemRefType>(fnType.getResult(0));
  if (!resultType) {
    reason = "single function result is not a memref type";
    return failure();
  }

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  if (returns.size() != 1) {
    reason = "function does not have exactly one func.return";
    return failure();
  }

  func::ReturnOp ret = returns.front();
  if (ret.getNumOperands() != 1) {
    reason = "func.return operand count is not exactly 1";
    return failure();
  }

  Value returnedValue = ret.getOperand(0);
  if (returnedValue.getType() != resultType) {
    reason = "func.return operand type does not match function result type";
    return failure();
  }

  auto alloc = returnedValue.getDefiningOp<memref::AllocOp>();
  if (!alloc) {
    reason = "returned memref is not directly produced by memref.alloc";
    return failure();
  }

  state = FunctionRewriteState{func, ret, alloc, returnedValue, resultType};
  return success();
}

void rewriteFunctionToOutParam(FunctionRewriteState &state) {
  func::FuncOp func = state.func;
  MLIRContext *ctx = func.getContext();

  FunctionType oldType = func.getFunctionType();
  SmallVector<Type> newInputs(oldType.getInputs().begin(), oldType.getInputs().end());
  newInputs.push_back(state.resultType);
  func.setType(FunctionType::get(ctx, newInputs, TypeRange{}));

  Block &entryBlock = func.front();
  BlockArgument outArg = entryBlock.addArgument(state.resultType, func.getLoc());

  // Copy to a mutable SSA value handle before RAUW.
  Value returnedValue = state.returnedValue;
  returnedValue.replaceAllUsesWith(outArg);
  state.returnOp->setOperands(ValueRange{});

  if (state.returnedAlloc->use_empty())
    state.returnedAlloc.erase();
}

} // namespace

namespace mlir {
namespace allo {

bool applyPYNQReturnMemrefToOutParam(ModuleOp &module) {
  llvm::StringSet<> rewrittenFunctionNames;

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.getFunctionType().getNumResults() == 0)
      continue;

    FunctionRewriteState state;
    std::string reason;
    if (failed(analyzeFunctionForRewrite(func, state, reason))) {
      func.emitRemark() << "[pynq-return-memref-to-out-param] skipped: "
                        << reason;
      continue;
    }

    rewriteFunctionToOutParam(state);
    rewrittenFunctionNames.insert(func.getName());
  }

  if (rewrittenFunctionNames.empty())
    return true;

  SmallVector<func::CallOp> callsToRewrite;
  module.walk([&](func::CallOp call) {
    if (rewrittenFunctionNames.contains(call.getCallee()))
      callsToRewrite.push_back(call);
  });

  bool hadError = false;
  for (func::CallOp call : callsToRewrite) {
    if (call.getNumResults() != 1) {
      call.emitError() << "[pynq-return-memref-to-out-param] call rewrite "
                          "failed: expected exactly 1 call result";
      hadError = true;
      continue;
    }

    auto memrefResultType = dyn_cast<MemRefType>(call.getResult(0).getType());
    if (!memrefResultType) {
      call.emitError() << "[pynq-return-memref-to-out-param] call rewrite "
                          "failed: call result is not memref";
      hadError = true;
      continue;
    }

    OpBuilder builder(call);
    Value outAlloc =
        builder.create<memref::AllocOp>(call.getLoc(), memrefResultType);

    SmallVector<Value> newOperands(call.getOperands().begin(),
                                   call.getOperands().end());
    newOperands.push_back(outAlloc);

    builder.create<func::CallOp>(call.getLoc(), call.getCallee(), TypeRange{},
                                 newOperands);

    call.getResult(0).replaceAllUsesWith(outAlloc);
    call.erase();
  }

  return !hadError;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloPYNQReturnMemrefToOutParam
    : public PYNQReturnMemrefToOutParamBase<AlloPYNQReturnMemrefToOutParam> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!applyPYNQReturnMemrefToOutParam(module))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>>
createPYNQReturnMemrefToOutParamPass() {
  return std::make_unique<AlloPYNQReturnMemrefToOutParam>();
}

} // namespace allo
} // namespace mlir
