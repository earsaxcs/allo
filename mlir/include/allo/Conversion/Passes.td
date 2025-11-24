/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MLIR_PASSES
#define ALLO_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def AlloToLLVMLowering : Pass<"allo-lower-to-llvm", "ModuleOp"> {
  let summary = "Allo to LLVM conversion pass";
  let constructor = "mlir::allo::createAlloToLLVMLoweringPass()";
}

def FixedToInteger : Pass<"fixed-to-integer", "ModuleOp"> {
  let summary = "Fixed-point operations to integer";
  let constructor = "mlir::allo::createFixedPointToIntegerPass()";
}

def LowerCompositeType : Pass<"lower-composite-type", "ModuleOp"> {
  let summary = "Lower composite types";
  let constructor = "mlir::allo::createLowerCompositeTypePass()";
}

def LowerBitOps : Pass<"lower-bit-ops", "ModuleOp"> {
  let summary = "Lower bit operations";
  let constructor = "mlir::allo::createLowerBitOpsPass()";
}

def LowerPrintOps : Pass<"lower-print-ops", "ModuleOp"> {
  let summary = "Lower print operations";
  let constructor = "mlir::allo::createLowerPrintOpsPass()";
}

def LowerAlloQuantToVivado : Pass<"lower-allo-quant-to-vivado", "ModuleOp"> {
  let summary = "Lower Allo quantized ops to Vivado backend ops";
  let description = [{
    This pass lowers platform-agnostic Allo quantization operations to 
    Vivado/PYNQ backend-specific operations. It performs a simple 1:1 mapping
    while preserving all mathematical semantics and adding default backend
    attributes for hardware implementation.
    
    Lowering patterns:
    - allo.qmatmul -> vivado.qmatmul (with default tiling and buffer allocation)
    - allo.qlinear -> vivado.qlinear (with bias fusion support)
    - allo.qadd -> vivado.qadd (with fusion hints)
    - allo.int_gelu -> vivado.int_gelu (with LUT implementation)
    - allo.int_softmax -> vivado.int_softmax (with LUT implementation)
    - allo.int_layernorm -> vivado.int_layernorm (with approx rsqrt)
  }];
  let constructor = "mlir::allo::createLowerAlloQuantToVivadoPass()";
  let dependentDialects = ["mlir::allo::vivado::VivadoDialect"];
}

#endif // ALLO_MLIR_PASSES