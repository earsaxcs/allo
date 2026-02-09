# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloPasses
    LoopTransformations.cpp
    AnyWidthInteger.cpp
    MoveReturnToInput.cpp
    Passes.cpp
    LegalizeCast.cpp
    RemoveStrideMap.cpp
    MemRefDCE.cpp
    DataPlacement.cpp
    UnifyKernels.cpp
    MergeSubviewAndCopy.cpp
    HoistGetGlobal.cpp
    PYNQHoistBufferAlloc.cpp
    PYNQSimplifyHostTransfers.cpp
    PYNQElideRedundantCopies.cpp
    PYNQScheduleOps.cpp
    PYNQBufferAllocation.cpp
    PYNQMidLower.cpp
    PYNQOptimizeSync.cpp
    PYNQOptimizeSubviewGlobals.cpp
    RepackVivadoScales.cpp
    VivadoMergeRedundantQuantDequant.cpp
    ToggleVivadoTranspose.cpp
    VivadoQLinearKSplit.cpp
    VivadoSeparateBias.cpp
    InsertScaleConversionAfterLayerNorm.cpp
    VivadoPadding.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/allo

    DEPENDS
    MLIRAlloOpsIncGen
    MLIRAlloTypesIncGen
    MLIRAlloPassesIncGen
    MLIRPYNQOpsIncGen
    MLIRPYNQTypesIncGen
    MLIRVivadoOpsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAllo
    MLIRAlloSupport
)