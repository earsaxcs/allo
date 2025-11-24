/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Dialect/VivadoDialect.h"
#include "allo/Dialect/VivadoOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::allo::vivado;

//===----------------------------------------------------------------------===//
// Vivado Dialect
//===----------------------------------------------------------------------===//

#include "allo/Dialect/VivadoDialect.cpp.inc"

void VivadoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "allo/Dialect/VivadoOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "allo/Dialect/VivadoOps.cpp.inc"
