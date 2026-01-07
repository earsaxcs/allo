/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef Allo_MLIR_PASSDETAIL_H
#define Allo_MLIR_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "allo/Dialect/VivadoDialect.h"
#include "allo/Dialect/PYNQDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace allo {

#define GEN_PASS_CLASSES
#include "allo/Transforms/Passes.h.inc"

} // namespace allo
} // end namespace mlir

#endif // Allo_MLIR_PASSDETAIL_H
