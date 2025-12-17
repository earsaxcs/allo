/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * EmitVivadoC - PYNQ Op to C Code Emission
 * 
 * This file implements the translation from PYNQ dialect operations to C code.
 * The generated C code calls runtime functions defined in pynq_runtime.h/c
 * that interface with the PYNQ accelerator hardware.
 * 
 * Supported PYNQ Operations:
 *   - pynq.matmul_instr: Matrix multiplication instruction -> exec_matmul()
 *   - pynq.data_transfer: DMA data transfer -> exec_data_load() / exec_data_store()
 *   - pynq.buffer_alloc: On-chip buffer allocation (compile-time)
 *   - pynq.sync: Synchronization barrier -> pynq_sync()
 *   - pynq.gelu: GELU activation -> exec_gelu()
 *   - pynq.qadd: Quantized add -> exec_qadd()
 *   - pynq.softmax: Softmax -> exec_softmax()
 *   - pynq.layernorm: Layer normalization -> exec_layernorm()
 *   - pynq.vector_op: Generic vector operation -> exec_vector()
 * 
 * Code Generation Pattern:
 *   The emitter generates calls to high-level runtime functions instead of
 *   low-level instruction encoding functions. The runtime layer (pynq_runtime.c)
 *   handles instruction generation, register writes, and error checking.
 * 
 * Generated C code example:
 *   #include "pynq_runtime.h"
 *   
 *   void kernel(...) {
 *       // Buffer allocation (resolved at compile time)
 *       // pynq.buffer_alloc: buffer_id=0, capacity=4096 bytes
 *       
 *       // Data transfer (host to device)
 *       exec_data_load(input_data, 0, 4, 128);
 *       
 *       // Matrix multiplication
 *       uint32_t buffer_msg[3] = {0, 1, 2};
 *       exec_matmul(buffer_msg, 4, 4, 128, 0, 0, 0, 0);
 *       
 *       // Vector operations
 *       exec_softmax(2, 4, 32);
 *       
 *       // Synchronization
 *       pynq_sync();
 *       
 *       // Data transfer (device to host)
 *       exec_data_store(output_data, 2, 4, 128);
 *   }
 */

#include "allo/Translation/EmitVivadoC.h"
#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Dialect/PYNQAttrs.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/Visitor.h"
#include "allo/Support/Utils.h"
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Value.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace allo;
using namespace mlir::allo::pynq;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Get C type name for a given MLIR type
static SmallString<16> getCTypeName(Type valType) {
  if (auto arrayType = valType.dyn_cast<ShapedType>())
    valType = arrayType.getElementType();

  // Handle integer types
  if (valType.isa<IndexType>())
    return SmallString<16>("int");
  else if (auto intType = valType.dyn_cast<IntegerType>()) {
    std::string signedness = "";
    if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
      signedness = "u";
    switch (intType.getWidth()) {
    case 1:
      return SmallString<16>("bool");
    case 8:
      return SmallString<16>(signedness + "int8_t");
    case 16:
      return SmallString<16>(signedness + "int16_t");
    case 32:
      return SmallString<16>(signedness + "int32_t");
    case 64:
      return SmallString<16>(signedness + "int64_t");
    default:
      return SmallString<16>(signedness + "int" +
                             std::to_string(intType.getWidth()) + "_t");
    }
  }
  // Handle float types
  else if (valType.isa<Float32Type>())
    return SmallString<16>("float");
  else if (valType.isa<Float64Type>())
    return SmallString<16>("double");

  return SmallString<16>("void");
}

static SmallString<16> getCTypeName(Value val) {
  return getCTypeName(val.getType());
}

//===----------------------------------------------------------------------===//
// PYNQCEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
using namespace mlir::allo::pynq;

/// Main emitter class for PYNQ operations to C code
class PYNQCEmitter : public AlloEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit PYNQCEmitter(AlloEmitterState &state) : AlloEmitterBase(state) {}

  //===--------------------------------------------------------------------===//
  // PYNQ Operation Emitters
  //===--------------------------------------------------------------------===//
  
  /// Emit pynq.matmul_instr operation
  void emitMatMulInstr(pynq::MatMulInstrOp op);
  
  /// Emit pynq.data_transfer operation
  void emitDataTransfer(pynq::DataTransferOp op);
  
  /// Emit pynq.buffer_alloc operation
  void emitBufferAlloc(pynq::BufferAllocOp op);
  
  /// Emit pynq.sync operation
  void emitSync(pynq::SyncOp op);
  
  /// Emit pynq.gelu operation
  void emitGELU(pynq::GELUOp op);
  
  /// Emit pynq.qadd operation
  void emitQAdd(pynq::QAddOp op);
  
  /// Emit pynq.softmax operation
  void emitSoftmax(pynq::SoftmaxOp op);
  
  /// Emit pynq.layernorm operation
  void emitLayerNorm(pynq::LayerNormOp op);
  
  /// Emit pynq.vector_op operation (generic)
  void emitVectorOp(pynq::VectorOp op);

  //===--------------------------------------------------------------------===//
  // Standard Operation Emitters (from Vivado HLS emitter)
  //===--------------------------------------------------------------------===//
  
  /// Emit memref operations
  void emitAlloc(memref::AllocOp op);
  void emitAlloc(memref::AllocaOp op);
  void emitDealloc(memref::DeallocOp op);
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);
  void emitGetGlobal(memref::GetGlobalOp op);
  void emitGlobal(memref::GlobalOp op);
  
  /// Emit function-related operations
  void emitCall(func::CallOp op);
  void emitReturn(func::ReturnOp op);
  
  /// Emit control flow operations
  void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitAffineFor(AffineForOp op);
  void emitAffineIf(AffineIfOp op);
  
  /// Emit constant operations
  void emitConstant(arith::ConstantOp op);
  
  //===--------------------------------------------------------------------===//
  // Top-level Module Emitter
  //===--------------------------------------------------------------------===//
  
  /// Emit the entire module
  void emitModule(ModuleOp module);

private:
  //===--------------------------------------------------------------------===//
  // Helper Methods
  //===--------------------------------------------------------------------===//
  
  /// Emit a value reference or declaration
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  
  /// Emit an array declaration
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "");
  
  /// Emit nested loop head for array operations
  unsigned emitNestedLoopHead(Value val);
  
  /// Emit nested loop tail
  void emitNestedLoopTail(unsigned rank);
  
  /// Emit line information comment
  void emitInfoAndNewLine(Operation *op);
  
  /// Emit a block of operations
  void emitBlock(Block &block);
  
  /// Emit a function definition
  void emitFunction(func::FuncOp func);
  
  /// Emit the C file header with includes
  void emitHeader();
  
  /// Emit intrinsic function declarations
  void emitIntrinsicDeclarations();
};
} // namespace

//===----------------------------------------------------------------------===//
// PYNQ Visitor Class
//===----------------------------------------------------------------------===//

namespace {
/// Visitor for dispatching PYNQ operations
class PYNQOpVisitor {
public:
  PYNQOpVisitor(PYNQCEmitter &emitter) : emitter(emitter) {}

  /// Dispatch to the appropriate emitter based on operation type
  bool dispatch(Operation *op) {
    // PYNQ operations
    if (auto matmulOp = dyn_cast<pynq::MatMulInstrOp>(op)) {
      emitter.emitMatMulInstr(matmulOp);
      return true;
    }
    if (auto transferOp = dyn_cast<pynq::DataTransferOp>(op)) {
      emitter.emitDataTransfer(transferOp);
      return true;
    }
    if (auto allocOp = dyn_cast<pynq::BufferAllocOp>(op)) {
      emitter.emitBufferAlloc(allocOp);
      return true;
    }
    if (auto syncOp = dyn_cast<pynq::SyncOp>(op)) {
      emitter.emitSync(syncOp);
      return true;
    }
    if (auto geluOp = dyn_cast<pynq::GELUOp>(op)) {
      emitter.emitGELU(geluOp);
      return true;
    }
    if (auto qaddOp = dyn_cast<pynq::QAddOp>(op)) {
      emitter.emitQAdd(qaddOp);
      return true;
    }
    if (auto softmaxOp = dyn_cast<pynq::SoftmaxOp>(op)) {
      emitter.emitSoftmax(softmaxOp);
      return true;
    }
    if (auto layernormOp = dyn_cast<pynq::LayerNormOp>(op)) {
      emitter.emitLayerNorm(layernormOp);
      return true;
    }
    if (auto vectorOp = dyn_cast<pynq::VectorOp>(op)) {
      emitter.emitVectorOp(vectorOp);
      return true;
    }
    
    // Standard operations
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      emitter.emitScfFor(forOp);
      return true;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      emitter.emitScfIf(ifOp);
      return true;
    }
    if (auto affineForOp = dyn_cast<AffineForOp>(op)) {
      emitter.emitAffineFor(affineForOp);
      return true;
    }
    if (auto affineIfOp = dyn_cast<AffineIfOp>(op)) {
      emitter.emitAffineIf(affineIfOp);
      return true;
    }
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      emitter.emitAlloc(allocOp);
      return true;
    }
    if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
      emitter.emitAlloc(allocaOp);
      return true;
    }
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      emitter.emitLoad(loadOp);
      return true;
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      emitter.emitStore(storeOp);
      return true;
    }
    if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(op)) {
      emitter.emitGetGlobal(getGlobalOp);
      return true;
    }
    if (auto globalOp = dyn_cast<memref::GlobalOp>(op)) {
      emitter.emitGlobal(globalOp);
      return true;
    }
    if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      emitter.emitDealloc(deallocOp);
      return true;
    }
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      emitter.emitCall(callOp);
      return true;
    }
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      emitter.emitReturn(returnOp);
      return true;
    }
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      emitter.emitConstant(constantOp);
      return true;
    }
    
    // Skip certain operations that don't need emission
    if (isa<scf::YieldOp>(op) || isa<AffineYieldOp>(op)) {
      return true;
    }
    
    return false;
  }

private:
  PYNQCEmitter &emitter;
};
} // namespace

//===----------------------------------------------------------------------===//
// PYNQCEmitter Implementation - PYNQ Operations
//===----------------------------------------------------------------------===//

void PYNQCEmitter::emitMatMulInstr(pynq::MatMulInstrOp op) {
  // First emit buffer array initialization
  indent();
  os << "uint32_t buffer_msg[3] = {";
  emitValue(op.getInputBufferId());
  os << ", ";
  emitValue(op.getWeightBufferId());
  os << ", ";
  emitValue(op.getOutputBufferId());
  os << "};\n";
  
  // Call exec_matmul from pynq_runtime.c
  indent();
  os << "exec_matmul(buffer_msg, ";
  emitValue(op.getInputTileCount());
  os << ", ";
  emitValue(op.getWeightTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ", ";
  emitValue(op.getHeadCount());
  os << ", ";
  emitValue(op.getHeadTileAxis());
  os << ", ";
  emitValue(op.getEnableBias());
  os << ", ";
  emitValue(op.getEnableTranspose());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitDataTransfer(pynq::DataTransferOp op) {
  // Generate call to exec_data_load (direction=0) or exec_data_store (direction=1)
  // Runtime functions handle both instruction generation and register write
  
  // Try to get constant direction value
  auto directionValue = op.getDirection();
  
  // Check if direction is a constant
  if (auto constOp = directionValue.getDefiningOp<arith::ConstantOp>()) {
    auto directionAttr = constOp.getValue().dyn_cast<IntegerAttr>();
    if (directionAttr) {
      int64_t direction = directionAttr.getInt();
      
      indent();
      if (direction == 0) {
        os << "exec_data_load(";
      } else {
        os << "exec_data_store(";
      }
      
      // Emit data pointer (cast to appropriate type)
      os << "(void*)";
      emitValue(op.getData());
      os << ", ";
      
      // Emit buffer_id, tile_count, total_pkg_num
      emitValue(op.getBufferId());
      os << ", ";
      emitValue(op.getTileCount());
      os << ", ";
      emitValue(op.getTotalPkgNum());
      os << ");";
      emitInfoAndNewLine(op);
      return;
    }
  }
  
  // If direction is not constant, emit runtime conditional
  indent();
  os << "if (";
  emitValue(directionValue);
  os << " == 0) {\n";
  addIndent();
  indent();
  os << "exec_data_load((void*)";
  emitValue(op.getData());
  os << ", ";
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getTotalPkgNum());
  os << ");\n";
  reduceIndent();
  indent();
  os << "} else {\n";
  addIndent();
  indent();
  os << "exec_data_store((void*)";
  emitValue(op.getData());
  os << ", ";
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getTotalPkgNum());
  os << ");\n";
  reduceIndent();
  indent();
  os << "}";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitBufferAlloc(pynq::BufferAllocOp op) {
  // Buffer allocation is typically resolved at compile time
  // Emit as a comment or placeholder for tracking
  indent();
  os << "// pynq.buffer_alloc: buffer_id=" << op.getBufferId()
     << ", capacity=" << op.getCapacityBytes() << " bytes";
  emitInfoAndNewLine(op);
  
  // If the buffer result is used, we need to track it
  if (!op.getBuffer().use_empty()) {
    indent();
    os << "pynq_buffer_t ";
    emitValue(op.getBuffer());
    os << " = pynq_buffer_get(" << op.getBufferId() << ");";
    os << "\n";
  }
}

void PYNQCEmitter::emitSync(pynq::SyncOp op) {
  indent();
  os << "pynq_sync();";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitGELU(pynq::GELUOp op) {
  indent();
  os << "exec_gelu(";
  
  // GELU operation parameters
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitQAdd(pynq::QAddOp op) {
  indent();
  os << "exec_qadd(";
  
  // QAdd operation parameters
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ", ";
  emitValue(op.getExtraBufferId());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSoftmax(pynq::SoftmaxOp op) {
  indent();
  os << "exec_softmax(";
  
  // Softmax operation parameters
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitLayerNorm(pynq::LayerNormOp op) {
  indent();
  os << "exec_layernorm(";
  
  // LayerNorm operation parameters
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitVectorOp(pynq::VectorOp op) {
  indent();
  os << "exec_vector(";
  
  // Generic vector operation parameters
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ", ";
  emitValue(op.getOp());
  os << ", ";
  emitValue(op.getExtraBufferId());
  os << ");";
  emitInfoAndNewLine(op);
}

// Interleave and Deinterleave ops are commented out in PYNQOps.td
// Remove their implementations
/*
void PYNQCEmitter::emitInterleave(pynq::InterleaveOp op) {
  indent();
  
  // Get the interleave pattern
  auto pattern = op.getPattern();
  const char* patternStr = "PYNQ_INTERLEAVE_NONE";
  switch (pattern) {
    case InterleavePattern::none:
      patternStr = "PYNQ_INTERLEAVE_NONE";
      break;
    case InterleavePattern::row_major_tiled:
      patternStr = "PYNQ_INTERLEAVE_ROW_MAJOR_TILED";
      break;
    case InterleavePattern::col_major_tiled:
      patternStr = "PYNQ_INTERLEAVE_COL_MAJOR_TILED";
      break;
    case InterleavePattern::weight_interleave:
      patternStr = "PYNQ_INTERLEAVE_WEIGHT";
      break;
    case InterleavePattern::custom:
      patternStr = "PYNQ_INTERLEAVE_CUSTOM";
      break;
  }
  
  os << "pynq_interleave(";
  emitValue(op.getOutput());
  os << ", ";
  emitValue(op.getInput());
  os << ", " << patternStr;
  
  // Emit tile sizes if present
  if (auto tileSizes = op.getTileSizes()) {
    os << ", ";
    auto tileArray = tileSizes->getAsValueRange<IntegerAttr>();
    bool first = true;
    for (auto tile : tileArray) {
      if (!first) os << ", ";
      os << tile.getZExtValue();
      first = false;
    }
  }
  
  os << ");";
  emitInfoAndNewLine(op);
}

// InterleaveOp and DeinterleaveOp are commented out in PYNQOps.td
*/

//===----------------------------------------------------------------------===//
// PYNQCEmitter Implementation - Standard Operations
//===----------------------------------------------------------------------===//

void PYNQCEmitter::emitAlloc(memref::AllocOp op) {
  if (isDeclared(op.getResult()))
    return;
  
  auto memrefType = op.getType();
  if (!memrefType.hasStaticShape()) {
    emitError(op, "dynamic shape not supported");
    return;
  }
  
  // Calculate total size in bytes
  int64_t totalElements = 1;
  for (auto dim : memrefType.getShape()) {
    totalElements *= dim;
  }
  auto elementType = memrefType.getElementType();
  unsigned elementSizeInBits = elementType.getIntOrFloatBitWidth();
  unsigned elementSizeInBytes = (elementSizeInBits + 7) / 8;
  int64_t totalBytes = totalElements * elementSizeInBytes;
  
  // Get variable name for this allocation
  auto varName = addName(op.getResult(), false);
  auto sharedMemName = varName + "_shared";
  
  // Emit PYNQ_SHARED_MEMORY struct declaration
  indent();
  os << "PYNQ_SHARED_MEMORY " << sharedMemName << ";";
  emitInfoAndNewLine(op);
  
  // Emit PYNQ_allocatedSharedMemory call
  indent();
  os << "PYNQ_allocatedSharedMemory(&" << sharedMemName << ", " 
     << totalBytes << ", 1);\n";
  
  // Emit pointer cast to access the data
  indent();
  os << getCTypeName(elementType) << "* " << varName 
     << " = (" << getCTypeName(elementType) << "*)" 
     << sharedMemName << ".pointer;\n";
}

void PYNQCEmitter::emitAlloc(memref::AllocaOp op) {
  if (isDeclared(op.getResult()))
    return;
  
  auto memrefType = op.getType();
  if (!memrefType.hasStaticShape()) {
    emitError(op, "dynamic shape not supported");
    return;
  }
  
  // AllocaOp is stack allocation, use regular C array (not shared memory)
  indent();
  emitArrayDecl(op.getResult());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitDealloc(memref::DeallocOp op) {
  // Get the memref being deallocated
  auto memref = op.getMemref();
  
  // Check if this memref was declared (it should have been via emitAlloc)
  if (!isDeclared(memref)) {
    // If not declared, skip - it might be an argument or external reference
    indent();
    os << "// Warning: dealloc for undeclared memref";
    emitInfoAndNewLine(op);
    return;
  }
  
  // Check if the memref came from an AllocOp (not AllocaOp)
  // AllocOp uses shared memory, AllocaOp uses stack
  if (auto allocOp = memref.getDefiningOp<memref::AllocOp>()) {
    auto varName = getName(memref);
    auto sharedMemName = varName + "_shared";
    
    indent();
    os << "PYNQ_freeSharedMemory(&" << sharedMemName << ");";
    emitInfoAndNewLine(op);
  }
  // AllocaOp (stack allocation) does not need explicit free
}

void PYNQCEmitter::emitLoad(memref::LoadOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemRef());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitStore(memref::StoreOp op) {
  indent();
  emitValue(op.getMemRef());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitGetGlobal(memref::GetGlobalOp op) {
  indent();
  os << "// reference to global: " << op.getName();
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitGlobal(memref::GlobalOp op) {
  auto init_val = op.getInitialValue();
  if (!init_val.has_value())
    return;
  
  auto attr = init_val.value();
  if (auto denseAttr = attr.dyn_cast<DenseElementsAttr>()) {
    indent();
    auto arrayType = op.getType().cast<ShapedType>();
    auto type = arrayType.getElementType();
    
    if (op->hasAttr("constant")) {
      os << "const ";
    }
    os << getCTypeName(type);
    os << " " << op.getSymName();
    for (auto &shape : arrayType.getShape())
      os << "[" << shape << "]";
    os << " = {";

    unsigned elementIdx = 0;
    for (auto element : denseAttr.getValues<Attribute>()) {
      if (type.isF32()) {
        auto value = element.cast<FloatAttr>().getValue().convertToFloat();
        os << value;
      } else if (type.isF64()) {
        auto value = element.cast<FloatAttr>().getValue().convertToDouble();
        os << value;
      } else if (type.isInteger(1))
        os << element.cast<BoolAttr>().getValue();
      else if (type.isIntOrIndex())
        os << element.cast<IntegerAttr>().getValue();

      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
  }
}

void PYNQCEmitter::emitCall(func::CallOp op) {
  indent();
  
  // Handle return values
  if (op.getNumResults() > 0) {
    for (auto result : op.getResults()) {
      if (!isDeclared(result)) {
        emitValue(result);
        os << " = ";
      }
    }
  }
  
  os << op.getCallee() << "(";
  
  unsigned argIdx = 0;
  for (auto arg : op.getOperands()) {
    emitValue(arg);
    if (argIdx++ != op.getNumOperands() - 1)
      os << ", ";
  }
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitReturn(func::ReturnOp op) {
  if (op.getNumOperands() == 0)
    return;
  
  indent();
  os << "return";
  if (op.getNumOperands() > 0) {
    os << " ";
    emitValue(op.getOperand(0));
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitScfFor(scf::ForOp op) {
  // Handle iter_args (loop-carried values)
  // For scf.for with iter_args, we need to:
  // 1. Declare/initialize variables before the loop
  // 2. Update them at the end of each iteration (from scf.yield)
  auto iterArgs = op.getInitArgs();
  auto regionArgs = op.getRegionIterArgs();
  
  if (!iterArgs.empty()) {
    // Emit initialization for loop-carried variables
    for (auto [initVal, regionArg] : llvm::zip(iterArgs, regionArgs)) {
      indent();
      os << getCTypeName(regionArg) << " ";
      emitValue(regionArg);
      os << " = ";
      emitValue(initVal);
      os << ";";
      emitInfoAndNewLine(op);
    }
  }
  
  // Emit the for loop header
  indent();
  os << "for (";
  auto iterVar = op.getInductionVar();
  
  // Loop initialization: declare and initialize the induction variable
  os << getCTypeName(iterVar) << " ";
  emitValue(iterVar);
  os << " = ";
  emitValue(op.getLowerBound());
  os << "; ";
  
  // Loop condition
  emitValue(iterVar);
  os << " < ";
  emitValue(op.getUpperBound());
  os << "; ";
  
  // Loop step
  emitValue(iterVar);
  os << " += ";
  emitValue(op.getStep());
  os << ") {";
  emitInfoAndNewLine(op);
  
  addIndent();
  
  // Emit loop body
  emitBlock(*op.getBody());
  
  // Handle scf.yield: update loop-carried variables
  // The scf.yield operands should be assigned to the corresponding iter_args
  if (!iterArgs.empty()) {
    auto &bodyBlock = op.getRegion().front();
    auto yieldOp = dyn_cast<scf::YieldOp>(bodyBlock.getTerminator());
    if (yieldOp) {
      auto yieldOperands = yieldOp.getOperands();
      for (auto [yieldVal, regionArg] : llvm::zip(yieldOperands, regionArgs)) {
        indent();
        emitValue(regionArg);
        os << " = ";
        emitValue(yieldVal);
        os << ";";
        emitInfoAndNewLine(yieldOp);
      }
    }
  }
  
  reduceIndent();
  
  indent();
  os << "}\n";
}

void PYNQCEmitter::emitScfIf(scf::IfOp op) {
  indent();
  os << "if (";
  emitValue(op.getCondition());
  os << ") {";
  emitInfoAndNewLine(op);
  
  addIndent();
  emitBlock(op.getThenRegion().front());
  reduceIndent();
  
  if (!op.getElseRegion().empty()) {
    indent();
    os << "} else {\n";
    addIndent();
    emitBlock(op.getElseRegion().front());
    reduceIndent();
  }
  
  indent();
  os << "}\n";
}

void PYNQCEmitter::emitAffineFor(AffineForOp op) {
  indent();
  auto iterVar = op.getInductionVar();
  
  os << "for (";
  emitValue(iterVar);
  os << " = " << op.getConstantLowerBound() << "; ";
  emitValue(iterVar);
  os << " < " << op.getConstantUpperBound() << "; ";
  emitValue(iterVar);
  if (op.getStep() == 1)
    os << "++) {";
  else
    os << " += " << op.getStep() << ") {";
  emitInfoAndNewLine(op);
  
  addIndent();
  emitBlock(*op.getBody());
  reduceIndent();
  
  indent();
  os << "}\n";
}

void PYNQCEmitter::emitAffineIf(AffineIfOp op) {
  indent();
  os << "if (/* affine condition */) {";
  emitInfoAndNewLine(op);
  
  addIndent();
  emitBlock(*op.getThenBlock());
  reduceIndent();
  
  if (op.hasElse()) {
    indent();
    os << "} else {\n";
    addIndent();
    emitBlock(*op.getElseBlock());
    reduceIndent();
  }
  
  indent();
  os << "}\n";
}

void PYNQCEmitter::emitConstant(arith::ConstantOp op) {
  // Scalar constants are emitted inline, so we don't need to do anything here
  // unless it's an array constant
  if (auto denseAttr = op.getValue().dyn_cast<DenseElementsAttr>()) {
    if (isDeclared(op.getResult()))
      return;
    
    indent();
    emitArrayDecl(op.getResult());
    os << " = {";
    
    auto type = op.getResult().getType().cast<ShapedType>().getElementType();
    unsigned elementIdx = 0;
    for (auto element : denseAttr.getValues<Attribute>()) {
      if (type.isF32()) {
        os << element.cast<FloatAttr>().getValue().convertToFloat();
      } else if (type.isF64()) {
        os << element.cast<FloatAttr>().getValue().convertToDouble();
      } else if (type.isInteger(1))
        os << element.cast<BoolAttr>().getValue();
      else if (type.isIntOrIndex())
        os << element.cast<IntegerAttr>().getValue();

      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
  }
}

//===----------------------------------------------------------------------===//
// PYNQCEmitter Implementation - Helper Methods
//===----------------------------------------------------------------------===//

void PYNQCEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              std::string name) {
  // Check if value has been declared
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }
  
  // Check if it's a constant operation
  if (auto defOp = val.getDefiningOp()) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      auto constAttr = constOp.getValue();
      if (auto intAttr = constAttr.dyn_cast<IntegerAttr>()) {
        os << intAttr.getInt();
        return;
      }
      if (auto boolAttr = constAttr.dyn_cast<BoolAttr>()) {
        os << (boolAttr.getValue() ? "true" : "false");
        return;
      }
      if (auto floatAttr = constAttr.dyn_cast<FloatAttr>()) {
        os << floatAttr.getValueAsDouble();
        return;
      }
    }
  }
  
  // Declare new variable
  os << getCTypeName(val) << " ";
  if (isPtr)
    os << "*";
  
  if (name.empty())
    os << addName(val, isPtr);
  else
    os << addName(val, isPtr, name);
  
  for (unsigned i = 0; i < rank; ++i)
    os << "[iv" << i << "]";
}

void PYNQCEmitter::emitArrayDecl(Value array, bool isFunc, std::string name) {
  if (isDeclared(array)) {
    os << getName(array);
    return;
  }
  
  auto arrayType = array.getType().cast<ShapedType>();
  if (arrayType.hasStaticShape()) {
    emitValue(array, 0, false, name);
    for (auto &shape : arrayType.getShape())
      os << "[" << shape << "]";
  } else {
    emitValue(array, 0, true, name);
  }
}

unsigned PYNQCEmitter::emitNestedLoopHead(Value val) {
  unsigned rank = 0;
  if (auto type = val.getType().dyn_cast<ShapedType>()) {
    if (!type.hasStaticShape())
      return 0;
    
    if (!isDeclared(val)) {
      indent();
      emitArrayDecl(val);
      os << ";\n";
    }
    
    unsigned dimIdx = 0;
    for (auto &shape : type.getShape()) {
      indent();
      os << "for (int iv" << dimIdx << " = 0; ";
      os << "iv" << dimIdx << " < " << shape << "; ";
      os << "++iv" << dimIdx++ << ") {\n";
      addIndent();
    }
    rank = type.getRank();
  }
  return rank;
}

void PYNQCEmitter::emitNestedLoopTail(unsigned rank) {
  for (unsigned i = 0; i < rank; ++i) {
    reduceIndent();
    indent();
    os << "}\n";
  }
}

void PYNQCEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
    os << " L" << loc.getLine();
  os << "\n";
}

void PYNQCEmitter::emitBlock(Block &block) {
  PYNQOpVisitor visitor(*this);
  for (auto &op : block) {
    if (!visitor.dispatch(&op)) {
      // Operation not handled, emit a comment
      indent();
      os << "// Unhandled operation: " << op.getName().getStringRef().str() << "\n";
    }
  }
}

void PYNQCEmitter::emitFunction(func::FuncOp func) {
  if (func.getBlocks().empty())
    return;
  
  if (func->hasAttr("top"))
    os << "/// This is the top function.\n";
  
  // Emit function signature
  os << "void " << func.getName() << "(\n";
  addIndent();
  
  // Emit arguments
  unsigned argIdx = 0;
  for (auto &arg : func.getArguments()) {
    indent();
    if (arg.getType().isa<ShapedType>())
      emitArrayDecl(arg, true);
    else
      emitValue(arg);
    
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }
  
  // Emit return type (for results)
  if (auto funcReturn = dyn_cast<func::ReturnOp>(func.front().getTerminator())) {
    for (auto result : funcReturn.getOperands()) {
      auto args = func.getArguments();
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        if (func.getNumArguments() > 0)
          os << ",\n";
        indent();
        if (result.getType().isa<ShapedType>())
          emitArrayDecl(result, true);
        else
          emitValue(result, 0, true);
      }
    }
  }
  
  reduceIndent();
  os << "\n) {\n";
  
  addIndent();
  emitBlock(func.front());
  reduceIndent();
  
  os << "}\n\n";
}

void PYNQCEmitter::emitHeader() {
  os << R"XXX(
//===------------------------------------------------------------*- C -*-===//
//
// Automatically generated C code for PYNQ accelerator.
// Generated from PYNQ dialect operations.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "instr.h"  // Hardware instruction generation functions
#include <pynq_api.h>  // PYNQ shared memory API

)XXX";
}

void PYNQCEmitter::emitIntrinsicDeclarations() {
  // The hardware instruction generation functions are declared in instr.h:
  // - uint32_t gen_matmul_instr(...) - Generate matrix multiplication instruction
  // - uint32_t gen_data_instr(...)   - Generate DMA transfer instruction
  // - uint32_t gen_vector_instr(...) - Generate vector operation instruction
  os << "// Hardware instruction functions declared in instr.h\n\n";
}

void PYNQCEmitter::emitModule(ModuleOp module) {
  emitHeader();
  emitIntrinsicDeclarations();
  
  // Emit global variables first
  for (auto &op : *module.getBody()) {
    if (auto globalOp = dyn_cast<memref::GlobalOp>(op))
      emitGlobal(globalOp);
  }
  
  // Then emit functions
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<func::FuncOp>(op))
      emitFunction(func);
  }
}

//===----------------------------------------------------------------------===//
// Entry Point for allo-translate
//===----------------------------------------------------------------------===//

LogicalResult allo::emitVivadoC(ModuleOp module, llvm::raw_ostream &os) {
  AlloEmitterState state(os);
  PYNQCEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void allo::registerEmitVivadoCTranslation() {
  static TranslateFromMLIRRegistration toVivadoC(
      "emit-vivado-c", "Emit C code for PYNQ accelerator", emitVivadoC,
      [&](DialectRegistry &registry) {
        registry.insert<
          mlir::allo::AlloDialect,
          mlir::allo::pynq::PYNQDialect,
          mlir::func::FuncDialect,
          mlir::arith::ArithDialect,
          mlir::tensor::TensorDialect,
          mlir::scf::SCFDialect,
          mlir::affine::AffineDialect,
          mlir::math::MathDialect,
          mlir::memref::MemRefDialect
        >();
      });
}
