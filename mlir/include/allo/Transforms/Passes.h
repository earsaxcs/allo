/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORMS_PASSES_H
#define ALLO_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/VivadoDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLoopTransformationPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createMoveReturnToInputPass();
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeCastPass();
std::unique_ptr<OperationPass<ModuleOp>> createRemoveStrideMapPass();
std::unique_ptr<OperationPass<ModuleOp>> createMemRefDCEPass();
std::unique_ptr<OperationPass<ModuleOp>> createDataPlacementPass();
std::unique_ptr<OperationPass<ModuleOp>> createMergeSubviewAndCopyPass();
std::unique_ptr<OperationPass<ModuleOp>> createHoistGetGlobalPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQHoistBufferAllocPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQSimplifyHostTransfersPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQElideRedundantCopiesPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQBufferAllocationPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQScheduleOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQAdjustScaleRShiftPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQMidLowerPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQReturnMemrefToOutParamPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQOptimizeSyncPass();
std::unique_ptr<OperationPass<ModuleOp>> createPYNQOptimizeSubviewGlobalsPass();
std::unique_ptr<OperationPass<ModuleOp>> createVivadoQLinearKSplitPass();
std::unique_ptr<OperationPass<ModuleOp>> createRepackVivadoScalesPass();
std::unique_ptr<OperationPass<ModuleOp>> createVivadoMergeRedundantQuantDequantPass();
std::unique_ptr<OperationPass<ModuleOp>> createToggleVivadoTransposePass();
std::unique_ptr<OperationPass<ModuleOp>> createVivadoPaddingPass();
std::unique_ptr<OperationPass<ModuleOp>> createVivadoSeparateBiasPass();
std::unique_ptr<OperationPass<ModuleOp>> createInsertScaleConversionAfterLayerNormPass();

bool applyLoopTransformation(ModuleOp &f);
bool applyAnyWidthInteger(ModuleOp &module);
bool applyMoveReturnToInput(ModuleOp &module);
bool applyLegalizeCast(ModuleOp &module);
bool applyRemoveStrideMap(ModuleOp &module);
bool applyMemRefDCE(ModuleOp &module);
bool applyDataPlacement(ModuleOp &module);
bool applyMergeSubviewAndCopy(ModuleOp &module, MLIRContext* ctxPtr);
bool applyPYNQReturnMemrefToOutParam(ModuleOp &module);
bool applyRepackVivadoScales(ModuleOp &module, MLIRContext *context);
bool applyVivadoMergeRedundantQuantDequant(ModuleOp &module, MLIRContext *context);
bool applyToggleVivadoTranspose(ModuleOp &module, MLIRContext *context);
bool applyVivadoPadding(ModuleOp &module, MLIRContext *context);
bool applyVivadoSeparateBias(ModuleOp &module, MLIRContext *context);
bool applyInsertScaleConversionAfterLayerNorm(ModuleOp &module, MLIRContext *context);
ModuleOp applyUnifyKernels(ModuleOp &module1, ModuleOp &module2, int loop_num);
// deprecated
// bool applyVivadoQLinearKSplit(ModuleOp &module, MLIRContext *context);

/// Registers all Allo transformation passes
void registerAlloPasses();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSFORMS_PASSES_H