/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_VIVADO_OPS_H
#define ALLO_VIVADO_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "allo/Dialect/VivadoDialect.h"

#define GET_OP_CLASSES
#include "allo/Dialect/VivadoOps.h.inc"

#endif // ALLO_VIVADO_OPS_H
