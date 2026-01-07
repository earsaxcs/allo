/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_CONVERSION_PASSES_H
#define ALLO_CONVERSION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "allo/Dialect/VivadoDialect.h"
#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/AlloOps.h"

namespace mlir {
namespace allo {

// Allo Dialect -> LLVM Dialect
std::unique_ptr<OperationPass<ModuleOp>> createAlloToLLVMLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createFixedPointToIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerCompositeTypePass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerPrintOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerAlloQuantToVivadoPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerVivadoToPYNQPass();

bool applyAlloToLLVMLoweringPass(ModuleOp &module, MLIRContext &context);
bool applyFixedPointToInteger(ModuleOp &module);
bool applyLowerCompositeType(ModuleOp &module);
bool applyLowerBitOps(ModuleOp &module);
bool applyLowerPrintOps(ModuleOp &module);
bool applyLowerAlloQuantToVivado(ModuleOp &module, MLIRContext *context);
bool applyLowerVivadoToPYNQ(ModuleOp &module, MLIRContext *context);

/// Registers all Allo conversion passes
void registerAlloConversionPasses();

#define GEN_PASS_CLASSES
#include "allo/Conversion/Passes.h.inc"

} // namespace allo
} // namespace mlir

#endif // ALLO_CONVERSION_PASSES_H