/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * PYNQ Dialect Attributes
 * 
 * This file defines attributes for the PYNQ dialect, including:
 * - Interleave patterns for data layout transformation
 * - Data source classification (static vs dynamic)
 */

#ifndef PYNQ_ATTRS
#define PYNQ_ATTRS

include "allo/Dialect/PYNQDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Interleave Pattern Enum
//===----------------------------------------------------------------------===//

// Defines different interleave patterns for data layout transformation
// These patterns determine how data is rearranged for optimal hardware access
// def PYNQ_InterleavePattern_None : I32EnumAttrCase<"none", 0, "none">;
// def PYNQ_InterleavePattern_RowMajorTiled : I32EnumAttrCase<"row_major_tiled", 1, "row_major_tiled">;
// def PYNQ_InterleavePattern_ColMajorTiled : I32EnumAttrCase<"col_major_tiled", 2, "col_major_tiled">;
// def PYNQ_InterleavePattern_WeightInterleave : I32EnumAttrCase<"weight_interleave", 3, "weight_interleave">;
// def PYNQ_InterleavePattern_Custom : I32EnumAttrCase<"custom", 4, "custom">;

// def PYNQ_InterleavePatternAttr : I32EnumAttr<
//     "InterleavePattern",
//     "Data interleave pattern for hardware-friendly layout",
//     [
//       PYNQ_InterleavePattern_None,
//       PYNQ_InterleavePattern_RowMajorTiled,
//       PYNQ_InterleavePattern_ColMajorTiled,
//       PYNQ_InterleavePattern_WeightInterleave,
//       PYNQ_InterleavePattern_Custom
//     ]> {
//   let cppNamespace = "::mlir::allo::pynq";
// }

//===----------------------------------------------------------------------===//
// Data Source Enum (for vivado op lowering decisions)
//===----------------------------------------------------------------------===//

// Classifies data sources to determine interleave timing
// def PYNQ_DataSource_Static : I32EnumAttrCase<"_static", 0, "_static">;
// def PYNQ_DataSource_Dynamic : I32EnumAttrCase<"_dynamic", 1, "_dynamic">;

// def PYNQ_DataSourceAttr : I32EnumAttr<
//     "DataSource",
//     "Classification of data source for interleave timing decisions",
//     [
//       PYNQ_DataSource_Static,   // Compile-time known (weights) -> offline interleave
//       PYNQ_DataSource_Dynamic   // Runtime data (activations) -> runtime interleave
//     ]> {
//   let cppNamespace = "::mlir::allo::pynq";
// }

//===----------------------------------------------------------------------===//
// Static Interleave Attribute (for memref.global marking)
//===----------------------------------------------------------------------===//

// This attribute is attached to memref.global to indicate that the data
// should be interleaved at compile/export time. The export tool reads this
// attribute and transforms the data before writing to binary files.
//
// Usage:
//   memref.global @weights : memref<128x128xi8> = dense<...>
//       {pynq.static_interleave = #pynq.static_interleave<weight_interleave>}

// def PYNQ_StaticInterleaveAttr : AttrDef<PYNQ_Dialect, "StaticInterleave"> {
//   let summary = "Marks static data for compile-time interleaving";
//   let description = [{
//     Attached to memref.global operations to indicate that the constant data
//     should be transformed to an interleaved layout during compilation/export.
    
//     The export tool (not the MLIR compiler) is responsible for:
//     1. Reading this attribute from the IR
//     2. Applying the specified interleave pattern to the data
//     3. Writing the transformed data to the output binary
    
//     This avoids runtime overhead for static data like weights.
    
//     Example:
//       memref.global @weights : memref<128x128xi8> = dense<...>
//           {pynq.static_interleave = #pynq.static_interleave<weight_interleave>}
//   }];
  
//   let mnemonic = "static_interleave";
  
//   let parameters = (ins
//     "InterleavePattern":$pattern
//   );
  
//   let assemblyFormat = "`<` $pattern `>`";
// }

#endif // PYNQ_ATTRS
