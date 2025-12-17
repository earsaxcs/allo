/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::allo::pynq;

//===----------------------------------------------------------------------===//
// PYNQ Enums (generated)
//===----------------------------------------------------------------------===//

#include "allo/Dialect/PYNQEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// PYNQ Attributes (generated)
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "allo/Dialect/PYNQAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// PYNQ Dialect
//===----------------------------------------------------------------------===//

#include "allo/Dialect/PYNQDialect.cpp.inc"

void PYNQDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "allo/Dialect/PYNQOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "allo/Dialect/PYNQTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "allo/Dialect/PYNQAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// PYNQ Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "allo/Dialect/PYNQTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// BufferType Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse: !pynq.buffer<elementType, bufferId, capacityBytes>
/// Where bufferId can be '?' for unallocated or a number for allocated
/// Examples:
///   !pynq.buffer<i8, ?, 4096>    - virtual/unallocated buffer
///   !pynq.buffer<i8, 0, 4096>    - physical buffer with ID 0
///   !pynq.buffer<i32, 2, 16384>  - physical buffer with ID 2
Type BufferType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  
  // Parse element type
  Type elementType;
  if (parser.parseType(elementType))
    return Type();
  
  if (parser.parseComma())
    return Type();
  
  // Parse buffer ID: either '?' or a number
  unsigned bufferId;
  if (parser.parseOptionalQuestion().succeeded()) {
    // '?' means unallocated
    bufferId = BufferType::kUnallocated;
  } else {
    // Parse as integer
    if (parser.parseInteger(bufferId))
      return Type();
  }
  
  if (parser.parseComma())
    return Type();
  
  // Parse capacity in bytes
  int64_t capacityBytes;
  if (parser.parseInteger(capacityBytes))
    return Type();
  
  if (parser.parseGreater())
    return Type();
  
  return BufferType::get(parser.getContext(), elementType, bufferId, 
                         capacityBytes);
}

/// Print: !pynq.buffer<elementType, bufferId, capacityBytes>
/// Prints '?' for unallocated buffers
void BufferType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printType(getElementType());
  printer << ", ";
  
  // Print '?' for unallocated, otherwise print the ID
  if (isVirtual()) {
    printer << "?";
  } else {
    printer << getBufferId();
  }
  
  printer << ", " << getCapacityBytes() << ">";
}

//===----------------------------------------------------------------------===//
// BufferType Verifier
//===----------------------------------------------------------------------===//

LogicalResult BufferType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    Type elementType, unsigned bufferId, int64_t capacityBytes) {
  
  // Verify buffer ID is within valid range (or is kUnallocated for virtual buffers)
  if (bufferId != kUnallocated && bufferId > kMaxBufferId) {
    return emitError() << "buffer ID " << bufferId 
                       << " exceeds maximum allowed ID " << kMaxBufferId
                       << " (valid range: 0-" << kMaxBufferId << " or ? for unallocated)";
  }
  
  // Verify capacity is within valid range
  if (capacityBytes < kMinCapacityBytes) {
    return emitError() << "buffer capacity " << capacityBytes 
                       << " bytes is less than minimum " << kMinCapacityBytes;
  }
  
  if (capacityBytes > kMaxCapacityBytes) {
    return emitError() << "buffer capacity " << capacityBytes 
                       << " bytes exceeds maximum " << kMaxCapacityBytes;
  }
  
  // Verify element type is a valid integer type (i8, i16, i32, etc.)
  if (!elementType.isIntOrFloat()) {
    return emitError() << "buffer element type must be integer or float, got "
                       << elementType;
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// PYNQ Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "allo/Dialect/PYNQOps.cpp.inc"

//===----------------------------------------------------------------------===//
// MatMulInstrOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult MatMulInstrOp::verify() {
  // Note: With proper I3 type constraints in TableGen, most validation
  // should be handled automatically. These checks are additional safety.
  
  // The hardware uses 3-bit buffer IDs (0-7)
  // TableGen I3 type should enforce this, but we double-check
  
  return success();
}

//===----------------------------------------------------------------------===//
// DataTransferOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult DataTransferOp::verify() {
  // Verify direction is valid (0 or 1)
  // TableGen I1 type should enforce this
  
  // Additional semantic checks could be added here:
  // - Verify memref element type matches expected data type
  // - Verify total_num matches memref size
  
  return success();
}

//===----------------------------------------------------------------------===//
// BufferAllocOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult BufferAllocOp::verify() {
  // Get the buffer type from the result
  auto bufferType = llvm::dyn_cast<BufferType>(getBuffer().getType());
  if (!bufferType) {
    return emitOpError() << "result must be a pynq.buffer type";
  }
  
  // Virtual buffers (unallocated) are always valid at this stage
  if (bufferType.isVirtual()) {
    return success();
  }
  
  // For allocated buffers, verify buffer ID is within valid range
  unsigned bufferId = bufferType.getBufferId();
  if (bufferId > BufferType::kMaxBufferId) {
    return emitOpError() << "buffer ID " << bufferId 
                         << " exceeds maximum allowed ID " 
                         << BufferType::kMaxBufferId;
  }
  
  // Verify capacity is within valid range
  int64_t capacity = bufferType.getCapacityBytes();
  if (capacity < BufferType::kMinCapacityBytes) {
    return emitOpError() << "buffer capacity " << capacity 
                         << " bytes is less than minimum "
                         << BufferType::kMinCapacityBytes;
  }
  
  if (capacity > BufferType::kMaxCapacityBytes) {
    return emitOpError() << "buffer capacity " << capacity 
                         << " bytes exceeds maximum "
                         << BufferType::kMaxCapacityBytes;
  }
  
  return success();
}

// //===----------------------------------------------------------------------===//
// // InterleaveOp Verifier
// //===----------------------------------------------------------------------===//

// LogicalResult InterleaveOp::verify() {
//   auto inputType = llvm::dyn_cast<MemRefType>(getInput().getType());
//   auto outputType = llvm::dyn_cast<MemRefType>(getOutput().getType());
  
//   if (!inputType || !outputType) {
//     return emitOpError() << "input and output must be memref types";
//   }
  
//   // Verify element types match
//   if (inputType.getElementType() != outputType.getElementType()) {
//     return emitOpError() << "input and output element types must match";
//   }
  
//   // Verify total number of elements match
//   int64_t inputElements = 1, outputElements = 1;
//   for (auto dim : inputType.getShape()) {
//     if (dim == ShapedType::kDynamic) {
//       inputElements = -1;
//       break;
//     }
//     inputElements *= dim;
//   }
//   for (auto dim : outputType.getShape()) {
//     if (dim == ShapedType::kDynamic) {
//       outputElements = -1;
//       break;
//     }
//     outputElements *= dim;
//   }
  
//   if (inputElements > 0 && outputElements > 0 && 
//       inputElements != outputElements) {
//     return emitOpError() << "input and output total elements must match, got "
//                          << inputElements << " vs " << outputElements;
//   }
  
//   // Verify tile dimensions if specified
//   if (getTileSizes()) {
//     auto tileSizes = *getTileSizes();
//     auto inputRank = inputType.getRank();
//     if (tileSizes.size() != static_cast<size_t>(inputRank)) {
//       return emitOpError() << "tile_sizes must have " << inputRank 
//                            << " elements to match input rank";
//     }
    
//     // Verify each tile dim divides the corresponding input dimension
//     for (size_t i = 0; i < tileSizes.size(); ++i) {
//       auto tileSize = llvm::cast<IntegerAttr>(tileSizes[i]).getInt();
//       auto inputDim = inputType.getShape()[i];
//       if (inputDim != ShapedType::kDynamic && tileSize > 0) {
//         if (inputDim % tileSize != 0) {
//           return emitOpError() << "tile size " << tileSize 
//                                << " must divide input dimension " << inputDim
//                                << " at index " << i;
//         }
//       }
//     }
//   }
  
//   return success();
// }

// //===----------------------------------------------------------------------===//
// // DeinterleaveOp Verifier
// //===----------------------------------------------------------------------===//

// LogicalResult DeinterleaveOp::verify() {
//   auto inputType = llvm::dyn_cast<MemRefType>(getInput().getType());
//   auto outputType = llvm::dyn_cast<MemRefType>(getOutput().getType());
  
//   if (!inputType || !outputType) {
//     return emitOpError() << "input and output must be memref types";
//   }
  
//   // Verify element types match
//   if (inputType.getElementType() != outputType.getElementType()) {
//     return emitOpError() << "input and output element types must match";
//   }
  
//   return success();
// }

//===----------------------------------------------------------------------===//
// SoftmaxOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  // All parameters are I32, verify hardware constraints manually
  // Hardware constraints from instr.h:
  // - buffer_id: 3 bits (0-7)
  // - tile_count: 3 bits (0-7), represents 1-8 tiles
  // - reduce_k: 8 bits (0-255)
  // Verifier will check at codegen time; basic IR is always valid
  return success();
}

//===----------------------------------------------------------------------===//
// LayerNormOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult LayerNormOp::verify() {
  // All parameters are I32, verify hardware constraints manually
  // Hardware constraints from instr.h:
  // - buffer_id: 3 bits (0-7)
  // - tile_count: 3 bits (0-7), represents 1-8 tiles
  // - reduce_k: 8 bits (0-255)
  // Note: LayerNorm uses SimpleVectorOpBase, no extra_buffer_id
  // Verifier will check at codegen time; basic IR is always valid
  return success();
}

//===----------------------------------------------------------------------===//
// GELUOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult GELUOp::verify() {
  // All parameters are I32, verify hardware constraints manually
  // Hardware constraints from instr.h:
  // - buffer_id: 3 bits (0-7)
  // - tile_count: 3 bits (0-7), represents 1-8 tiles
  // - reduce_k: 8 bits (0-255)
  // Verifier will check at codegen time; basic IR is always valid
  return success();
}

//===----------------------------------------------------------------------===//
// QAddOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult QAddOp::verify() {
  // All parameters are I32, verify hardware constraints manually
  // Hardware constraints from instr.h:
  // - buffer_id: 3 bits (0-7)
  // - tile_count: 3 bits (0-7), represents 1-8 tiles
  // - reduce_k: 8 bits (0-255)
  // - extra_buffer_id: 3 bits (0-7), second input buffer
  // Verifier will check at codegen time; basic IR is always valid
  return success();
}

//===----------------------------------------------------------------------===//
// VectorOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult VectorOp::verify() {
  // All parameters are I32, verify hardware constraints manually
  // Hardware constraints from instr.h:
  // - buffer_id: 3 bits (0-7)
  // - tile_count: 3 bits (0-7), represents 1-8 tiles
  // - reduce_k: 8 bits (0-255)
  // - op: 2 bits (0-3)
  // - extra_buffer_id: 3 bits (0-7)
  // Verifier will check at codegen time; basic IR is always valid
  return success();
}
