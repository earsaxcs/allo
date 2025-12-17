/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * EmitVivadoC - PYNQ Op to C Code Emission
 * 
 * This header declares the translation interface for emitting C code
 * from PYNQ dialect operations. The generated C code uses intrinsic
 * functions that interface with the PYNQ accelerator hardware.
 */

#ifndef ALLO_TRANSLATION_EMITVIVADOC_H
#define ALLO_TRANSLATION_EMITVIVADOC_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace allo {

/// Emit C code for PYNQ operations in the given module.
/// The generated code uses PYNQ intrinsic functions defined in pynq_intrinsics.h
LogicalResult emitVivadoC(ModuleOp module, llvm::raw_ostream &os);

/// Register the emit-vivado-c translation with mlir-translate.
void registerEmitVivadoCTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITVIVADOC_H
