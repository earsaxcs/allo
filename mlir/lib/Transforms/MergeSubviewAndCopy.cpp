/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Support/Utils.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <map>
#include <set>

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

bool applyMergeSubviewAndCopy(ModuleOp &mod) {
  return true;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloMergeSubviewAndCopyTransformation : public MergeSubviewAndCopyBase<AlloMergeSubviewAndCopyTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyMergeSubviewAndCopy(mod))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createMergeSubviewAndCopyPass() {
    return std::make_unique<AlloMergeSubviewAndCopyTransformation>();
}

} // namespace allo
} // namespace mlir