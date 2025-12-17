/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * C API for EmitVivadoC (PYNQ to C code emission)
 */

#ifndef ALLO_C_TRANSLATION_EMITVIVADOC_H
#define ALLO_C_TRANSLATION_EMITVIVADOC_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emit C code for PYNQ operations from the given MLIR module.
/// The generated code uses PYNQ intrinsic functions.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitVivadoC(MlirModule module,
                                                     MlirStringCallback callback,
                                                     void *userData);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_TRANSLATION_EMITVIVADOC_H
