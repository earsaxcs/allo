/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VIVADO_DIALECT
#define VIVADO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Vivado dialect definition
//===----------------------------------------------------------------------===//

def Vivado_Dialect : Dialect {
  let name = "vivado";
  let summary = "Vivado HLS backend quantization dialect";
  let description = [{
    This dialect provides backend-specific quantized operations for Vivado HLS
    and PYNQ hardware. It serves as an intermediate representation between
    platform-agnostic Allo quantization ops and low-level PYNQ hardware instructions.
    
    Operations in this dialect:
    - Preserve full quantization semantics from Allo ops
    - Add backend-specific optimization attributes
    - Enable hardware-aware transformations (tiling, buffering, scheduling)
    - Support multiple PYNQ hardware configurations
    
    Target platforms: Xilinx PYNQ-Z1/Z2, Ultra96, ZCU104, etc.
  }];
  let cppNamespace = "::mlir::allo::vivado";
  // let useDefaultAttributePrinterParser = 1;
  // let useDefaultTypePrinterParser = 1;    
}

//===----------------------------------------------------------------------===//
// Base Vivado operation definition
//===----------------------------------------------------------------------===//

class Vivado_Op<string mnemonic, list<Trait> traits = []> :
    Op<Vivado_Dialect, mnemonic, traits>;

#endif // VIVADO_DIALECT
