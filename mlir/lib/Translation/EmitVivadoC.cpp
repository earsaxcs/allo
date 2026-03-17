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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Value.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
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

static SmallString<16> getUnsignedCIntTypeName(IntegerType intType) {
  switch (intType.getWidth()) {
  case 1:
    return SmallString<16>("bool");
  case 8:
    return SmallString<16>("uint8_t");
  case 16:
    return SmallString<16>("uint16_t");
  case 32:
    return SmallString<16>("uint32_t");
  case 64:
    return SmallString<16>("uint64_t");
  default:
    return SmallString<16>("uint" + std::to_string(intType.getWidth()) + "_t");
  }
}

static bool getStaticStridesAndOffset(MemRefType type,
                                      SmallVector<int64_t, 4> &strides,
                                      int64_t &offset) {
  if (failed(getStridesAndOffset(type, strides, offset)))
    return false;
  for (auto stride : strides) {
    if (stride == ShapedType::kDynamic)
      return false;
  }
  if (offset == ShapedType::kDynamic)
    return false;
  return true;
}

static bool isRowMajorContiguous(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> strides) {
  if (shape.size() != strides.size())
    return false;
  int64_t expected = 1;
  for (int64_t i = (int64_t)shape.size() - 1; i >= 0; --i) {
    if (strides[i] != expected)
      return false;
    expected *= shape[i];
  }
  return true;
}

static SmallVector<int64_t, 4> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> strides(shape.size(), 1);
  int64_t stride = 1;
  for (int64_t i = (int64_t)shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

static int64_t computeAlignedByteSize(int64_t elementCount,
                                      unsigned elementBytes) {
  int64_t totalBytes = elementCount * static_cast<int64_t>(elementBytes);
  if (totalBytes < 8)
    return totalBytes;
  return ((totalBytes + 7) / 8) * 8;
}

static bool writeDenseElementsToBinary(DenseElementsAttr denseAttr,
                                        Type elementType,
                                        StringRef filePath,
                                        std::string &errorMsg) {
  std::error_code ec;
  llvm::raw_fd_ostream output(filePath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    errorMsg = "failed to open binary output: " + filePath.str();
    return false;
  }

  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  unsigned byteWidth = (bitWidth + 7) / 8;

  auto writeAPIntBytes = [&](const APInt &value) {
    for (unsigned byteIdx = 0; byteIdx < byteWidth; ++byteIdx) {
      uint8_t byte =
          (uint8_t)value.extractBits(8, byteIdx * 8).getZExtValue();
      output.write(reinterpret_cast<const char *>(&byte), 1);
    }
  };

  for (auto element : denseAttr.getValues<Attribute>()) {
    if (elementType.isF32() || elementType.isF64()) {
      auto fp = element.cast<FloatAttr>().getValue();
      writeAPIntBytes(fp.bitcastToAPInt());
    } else if (elementType.isInteger(1)) {
      auto value = element.cast<BoolAttr>().getValue();
      APInt bits(1, value ? 1 : 0);
      writeAPIntBytes(bits);
    } else if (elementType.isIntOrIndex()) {
      auto intVal = element.cast<IntegerAttr>().getValue();
      writeAPIntBytes(intVal);
    } else {
      errorMsg = "array has unsupported element type for binary export";
      return false;
    }
  }

  int64_t actualBytes =
      denseAttr.getNumElements() * static_cast<int64_t>(byteWidth);
  int64_t alignedBytes =
      computeAlignedByteSize(denseAttr.getNumElements(), byteWidth);
  int64_t padBytes = alignedBytes - actualBytes;
  if (padBytes > 0) {
    uint8_t zero = 0;
    for (int64_t i = 0; i < padBytes; ++i)
      output.write(reinterpret_cast<const char *>(&zero), 1);
  }

  return true;
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
  
  /// Emit pynq.data_transfer_instr operation
  void emitDataTransferInstr(pynq::DataTransferInstrOp op);

  /// Emit pynq.copy operation (abstract data movement)
  void emitCopy(pynq::CopyOp op);

  /// Emit pynq.view operation (buffer logical view)
  void emitView(pynq::ViewOp op);
  
  /// Emit pynq.buffer_alloc operation
  void emitBufferAlloc(pynq::BufferAllocOp op);
  
  /// Emit pynq.sync operation
  void emitSync(pynq::SyncOp op);

  /// Emit pynq.setMagic operation
  void emitSetMagic(pynq::SetMagicOp op);
  
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
  void emitVectorInstr(pynq::VectorInstrOp op);
  
  /// Emit pynq.activation_layout_transpose operation
  void emitActivationLayoutTranspose(pynq::ActivationLayoutTransposeOp op);

  /// Emit pynq.activation_dynamic_pad operation
  void emitActivationDynamicPad(pynq::ActivationDynamicPadOp op);

  /// Emit pynq.quant operation
  void emitQuant(pynq::QuantOp op);

  /// Emit pynq.dequant operation
  void emitDequant(pynq::DequantOp op);

  //===--------------------------------------------------------------------===//
  // Standard Operation Emitters (from Vivado HLS emitter)
  //===--------------------------------------------------------------------===//
  
  /// Emit memref operations
  void emitAlloc(memref::AllocOp op);
  void emitAlloc(memref::AllocaOp op);
  void emitDealloc(memref::DeallocOp op);
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);
  void emitMemrefCopy(memref::CopyOp op);
  void emitCast(memref::CastOp op);
  void emitSubview(memref::SubViewOp op);
  void emitReinterpretCast(memref::ReinterpretCastOp op);
  void emitReshape(memref::ReshapeOp op);
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

  /// Emit arith operations (common scalar IR produced after lowering)
  void emitAddI(arith::AddIOp op);
  void emitSubI(arith::SubIOp op);
  void emitMulI(arith::MulIOp op);
  void emitDivSI(arith::DivSIOp op);
  void emitDivUI(arith::DivUIOp op);
  void emitRemSI(arith::RemSIOp op);
  void emitRemUI(arith::RemUIOp op);
  void emitAndI(arith::AndIOp op);
  void emitOrI(arith::OrIOp op);
  void emitXorI(arith::XOrIOp op);
  void emitShLI(arith::ShLIOp op);
  void emitShRSI(arith::ShRSIOp op);
  void emitShRUI(arith::ShRUIOp op);
  void emitAddF(arith::AddFOp op);
  void emitSubF(arith::SubFOp op);
  void emitMulF(arith::MulFOp op);
  void emitDivF(arith::DivFOp op);
  void emitNegF(arith::NegFOp op);
  void emitCmpI(arith::CmpIOp op);
  void emitCmpF(arith::CmpFOp op);
  void emitSelect(arith::SelectOp op);
  void emitExtSI(arith::ExtSIOp op);
  void emitExtUI(arith::ExtUIOp op);
  void emitTruncI(arith::TruncIOp op);
  void emitSIToFP(arith::SIToFPOp op);
  void emitUIToFP(arith::UIToFPOp op);
  void emitFPToSI(arith::FPToSIOp op);
  void emitFPToUI(arith::FPToUIOp op);
  void emitExtF(arith::ExtFOp op);
  void emitTruncF(arith::TruncFOp op);
  void emitIndexCast(arith::IndexCastOp op);
  void emitIndexCastUI(arith::IndexCastUIOp op);
  
  //===--------------------------------------------------------------------===//
  // Top-level Module Emitter
  //===--------------------------------------------------------------------===//
  
  /// Emit the entire module
  void emitModule(ModuleOp module);

private:
  //===--------------------------------------------------------------------===//
  // Helper Methods
  //===--------------------------------------------------------------------===//
  
  /// Emit a C declarator for a memref represented as a pointer.
  ///
  /// For rank <= 1 (or dynamic shapes), emits `T* name`.
  /// For rank > 1 with static shape [d0, d1, ...], emits `T (*name)[d1]...[dN]`.
  void emitMemrefVarDecl(MemRefType memrefType, StringRef name);
  
  /// Emit the corresponding pointer type used for casting.
  ///
  /// For rank <= 1 (or dynamic shapes), emits `T*`.
  /// For rank > 1 with static shape [d0, d1, ...], emits `T (*)[d1]...[dN]`.
  void emitMemrefPtrCastType(MemRefType memrefType);
  
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

  /// Emit a numeric buffer ID from a pynq.buffer typed value.
  /// Falls back to emitting the value if it isn't a pynq.buffer.
  void emitBufferId(Value bufferLike);

  bool embedGlobalData = false;
  std::string globalBinDir = "pynq_global_bins";
  std::string globalParentDir = "vivado.prj";
  unsigned sharedMemTempCounter = 0;
  unsigned subviewTempCounter = 0;
  unsigned subviewPackCounter = 0;
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
    if (auto vectorInstrOp = dyn_cast<pynq::VectorInstrOp>(op)) {
      emitter.emitVectorInstr(vectorInstrOp);
      return true;
    }
    if (auto transferOp = dyn_cast<pynq::DataTransferInstrOp>(op)) {
      emitter.emitDataTransferInstr(transferOp);
      return true;
    }
    if (auto copyOp = dyn_cast<pynq::CopyOp>(op)) {
      emitter.emitCopy(copyOp);
      return true;
    }
    if (auto viewOp = dyn_cast<pynq::ViewOp>(op)) {
      emitter.emitView(viewOp);
      return true;
    }
    if (auto allocOp = dyn_cast<pynq::BufferAllocOp>(op)) {
      emitter.emitBufferAlloc(allocOp);
      return true;
    }
    if (auto setMagicOp = dyn_cast<pynq::SetMagicOp>(op)) {
      emitter.emitSetMagic(setMagicOp);
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
    if (auto transposeOp = dyn_cast<pynq::ActivationLayoutTransposeOp>(op)) {
      emitter.emitActivationLayoutTranspose(transposeOp);
      return true;
    }
    if (auto padOp = dyn_cast<pynq::ActivationDynamicPadOp>(op)) {
      emitter.emitActivationDynamicPad(padOp);
      return true;
    }
    if (auto quantOp = dyn_cast<pynq::QuantOp>(op)) {
      emitter.emitQuant(quantOp);
      return true;
    }
    if (auto dequantOp = dyn_cast<pynq::DequantOp>(op)) {
      emitter.emitDequant(dequantOp);
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
    if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      emitter.emitMemrefCopy(copyOp);
      return true;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(op)) {
      emitter.emitCast(castOp);
      return true;
    }
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      emitter.emitSubview(subviewOp);
      return true;
    }
    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
      emitter.emitReinterpretCast(reinterpretOp);
      return true;
    }
    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(op)) {
      emitter.emitReshape(reshapeOp);
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

    // arith dialect operations
    if (auto addiOp = dyn_cast<arith::AddIOp>(op)) {
      emitter.emitAddI(addiOp);
      return true;
    }
    if (auto subiOp = dyn_cast<arith::SubIOp>(op)) {
      emitter.emitSubI(subiOp);
      return true;
    }
    if (auto muliOp = dyn_cast<arith::MulIOp>(op)) {
      emitter.emitMulI(muliOp);
      return true;
    }
    if (auto divsiOp = dyn_cast<arith::DivSIOp>(op)) {
      emitter.emitDivSI(divsiOp);
      return true;
    }
    if (auto divuiOp = dyn_cast<arith::DivUIOp>(op)) {
      emitter.emitDivUI(divuiOp);
      return true;
    }
    if (auto remsiOp = dyn_cast<arith::RemSIOp>(op)) {
      emitter.emitRemSI(remsiOp);
      return true;
    }
    if (auto remuiOp = dyn_cast<arith::RemUIOp>(op)) {
      emitter.emitRemUI(remuiOp);
      return true;
    }
    if (auto andiOp = dyn_cast<arith::AndIOp>(op)) {
      emitter.emitAndI(andiOp);
      return true;
    }
    if (auto oriOp = dyn_cast<arith::OrIOp>(op)) {
      emitter.emitOrI(oriOp);
      return true;
    }
    if (auto xoriOp = dyn_cast<arith::XOrIOp>(op)) {
      emitter.emitXorI(xoriOp);
      return true;
    }
    if (auto shliOp = dyn_cast<arith::ShLIOp>(op)) {
      emitter.emitShLI(shliOp);
      return true;
    }
    if (auto shrsiOp = dyn_cast<arith::ShRSIOp>(op)) {
      emitter.emitShRSI(shrsiOp);
      return true;
    }
    if (auto shruiOp = dyn_cast<arith::ShRUIOp>(op)) {
      emitter.emitShRUI(shruiOp);
      return true;
    }
    if (auto addfOp = dyn_cast<arith::AddFOp>(op)) {
      emitter.emitAddF(addfOp);
      return true;
    }
    if (auto subfOp = dyn_cast<arith::SubFOp>(op)) {
      emitter.emitSubF(subfOp);
      return true;
    }
    if (auto mulfOp = dyn_cast<arith::MulFOp>(op)) {
      emitter.emitMulF(mulfOp);
      return true;
    }
    if (auto divfOp = dyn_cast<arith::DivFOp>(op)) {
      emitter.emitDivF(divfOp);
      return true;
    }
    if (auto negfOp = dyn_cast<arith::NegFOp>(op)) {
      emitter.emitNegF(negfOp);
      return true;
    }
    if (auto cmpiOp = dyn_cast<arith::CmpIOp>(op)) {
      emitter.emitCmpI(cmpiOp);
      return true;
    }
    if (auto cmpfOp = dyn_cast<arith::CmpFOp>(op)) {
      emitter.emitCmpF(cmpfOp);
      return true;
    }
    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      emitter.emitSelect(selectOp);
      return true;
    }
    if (auto extsiOp = dyn_cast<arith::ExtSIOp>(op)) {
      emitter.emitExtSI(extsiOp);
      return true;
    }
    if (auto extuiOp = dyn_cast<arith::ExtUIOp>(op)) {
      emitter.emitExtUI(extuiOp);
      return true;
    }
    if (auto trunciOp = dyn_cast<arith::TruncIOp>(op)) {
      emitter.emitTruncI(trunciOp);
      return true;
    }
    if (auto sitofpOp = dyn_cast<arith::SIToFPOp>(op)) {
      emitter.emitSIToFP(sitofpOp);
      return true;
    }
    if (auto uitofpOp = dyn_cast<arith::UIToFPOp>(op)) {
      emitter.emitUIToFP(uitofpOp);
      return true;
    }
    if (auto fptosiOp = dyn_cast<arith::FPToSIOp>(op)) {
      emitter.emitFPToSI(fptosiOp);
      return true;
    }
    if (auto fptouiOp = dyn_cast<arith::FPToUIOp>(op)) {
      emitter.emitFPToUI(fptouiOp);
      return true;
    }
    if (auto extfOp = dyn_cast<arith::ExtFOp>(op)) {
      emitter.emitExtF(extfOp);
      return true;
    }
    if (auto truncfOp = dyn_cast<arith::TruncFOp>(op)) {
      emitter.emitTruncF(truncfOp);
      return true;
    }
    if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(op)) {
      emitter.emitIndexCast(indexCastOp);
      return true;
    }
    if (auto indexCastUIOp = dyn_cast<arith::IndexCastUIOp>(op)) {
      emitter.emitIndexCastUI(indexCastUIOp);
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
  // Scope temporaries to avoid redeclaration when multiple matmul instrs
  // are emitted into the same C/C++ block.
  indent();
  os << "{\n";
  addIndent();

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
  emitValue(op.getHeadIdx());
  os << ", ";
  emitValue(op.getHeadTileAxis());
  os << ", ";
  emitValue(op.getEnableBias());
  os << ", ";
  emitValue(op.getEnableTranspose());
  os << ");";
  emitInfoAndNewLine(op);

  reduceIndent();
  indent();
  os << "}\n";
}

void PYNQCEmitter::emitDataTransferInstr(pynq::DataTransferInstrOp op) {
  // Generate call to exec_data_load (direction=0) / exec_data_store (direction=1)
  // or exec_data_to_fifo (direction=2/3).
  // Runtime functions handle both instruction generation and register write

  auto dataVal = op.getData();
  // NOTE: Data transfer is defined over the *data operand's* memref view.
  // We may still peel view-like wrappers only to find the backing allocation
  // for selecting the correct PYNQ_SHARED_MEMORY state.
  memref::SubViewOp subviewOp = dataVal.getDefiningOp<memref::SubViewOp>();

  // Try to get constant direction value early for subview handling.
  int64_t directionConst = -1;
  if (auto constOp = op.getDirection().getDefiningOp<arith::ConstantOp>()) {
    if (auto directionAttr = constOp.getValue().dyn_cast<IntegerAttr>())
      directionConst = directionAttr.getInt();
  }

  std::string sharedTempName;
  bool packedSubview = false;
  bool unpackAfterStore = false;
  MemRefType subviewResultType;
  SmallVector<int64_t, 4> subviewStrides;
  int64_t subviewOffset = 0;
  std::string subviewSourceName;
  bool stagingFromNormal = false;
  int64_t stagingFromNormalBytes = 0;
  std::string stagingFromNormalPtr;

  // Always analyze the data operand's view type for contiguity/offset.
  subviewResultType = dataVal.getType().cast<MemRefType>();
  if (!getStaticStridesAndOffset(subviewResultType, subviewStrides,
                                 subviewOffset)) {
    emitError(op, "dynamic memref stride/offset not supported for DMA");
    return;
  }

  // Resolve the element pointer name for packing/unpacking (backing buffer).
  Value indexingBase = dataVal;
  while (auto *def = indexingBase.getDefiningOp()) {
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      indexingBase = castOp.getSource();
      continue;
    }
    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
      indexingBase = reshapeOp.getSource();
      continue;
    }
    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
      indexingBase = reinterpretOp.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      indexingBase = subview.getSource();
      continue;
    }
    break;
  }
  subviewSourceName = getName(indexingBase).str().str();
  if (subviewSourceName.empty()) {
    emitError(op, "data operand must be declared before DMA packing");
    return;
  }

  // Compute the byte size of this transfer view.
  auto elementTypeForSize = subviewResultType.getElementType();
  int64_t totalElementsForSize = 1;
  for (auto dim : subviewResultType.getShape())
    totalElementsForSize *= dim;
  unsigned elementSizeBitsForSize = elementTypeForSize.getIntOrFloatBitWidth();
  unsigned elementSizeBytesForSize = (elementSizeBitsForSize + 7) / 8;
  int64_t totalBytesForSize = totalElementsForSize * elementSizeBytesForSize;

  bool contiguous =
      isRowMajorContiguous(subviewResultType.getShape(), subviewStrides);
  // NOTE: even if the view is contiguous, DMA cannot represent a non-zero
  // offset without changing the backing shared-memory physical address.
  // For offset!=0, stage via a contiguous pack/unpack buffer.
  if (!contiguous || subviewOffset != 0) {
    if (directionConst == 1) {
      unpackAfterStore = true;
    } else if (directionConst == 0 || directionConst == 2 ||
               directionConst == 3) {
      packedSubview = true;
    } else {
      emitError(op, "non-contiguous view requires constant direction for DMA");
      return;
    }

    auto elementType = subviewResultType.getElementType();
    int64_t totalBytes = totalBytesForSize;

    std::string packName =
        std::string("pynq_pack_sm") + std::to_string(subviewPackCounter++);
    indent();
    os << "PYNQ_SHARED_MEMORY " << packName << ";\n";
    indent();
    os << "PYNQ_allocatedSharedMemory(&" << packName << ", "
       << totalBytes << ", 1);\n";
    indent();
    os << getCTypeName(elementType) << "* " << packName << "_ptr = ("
       << getCTypeName(elementType) << "*)" << packName << ".pointer;\n";

     if (packedSubview) {
      indent();
      os << getCTypeName(elementType) << "* " << packName << "_src = ("
        << getCTypeName(elementType) << "*)" << subviewSourceName << ";\n";
     }

    if (packedSubview) {
      auto rowStrides =
          computeRowMajorStrides(subviewResultType.getShape());
      unsigned loopId = subviewTempCounter++;
      for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
        indent();
        os << "for (int sv" << loopId << "_" << i << " = 0; sv" << loopId
           << "_" << i << " < " << subviewResultType.getShape()[i]
           << "; ++sv" << loopId << "_" << i << ") {\n";
        addIndent();
      }

      indent();
      os << packName << "_ptr[";
      for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
        if (i != 0)
          os << " + ";
        os << "sv" << loopId << "_" << i << " * " << rowStrides[i];
      }
      os << "] = " << packName << "_src[" << subviewOffset;
      for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
        os << " + sv" << loopId << "_" << i << " * " << subviewStrides[i];
      }
      os << "];\n";

      for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
        reduceIndent();
        indent();
        os << "}\n";
      }
    }

    sharedTempName = packName;
  }

  if (sharedTempName.empty()) {
    // Direct DMA must use the real backing shared-memory state (<name>_shared).
    // Do not fabricate a PYNQ_SHARED_MEMORY by assigning to `.pointer`.
    // Find the ultimate backing value for selecting the shared-memory state.
    Value backing = dataVal;
    while (auto *def = backing.getDefiningOp()) {
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        backing = castOp.getSource();
        continue;
      }
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
        backing = reshapeOp.getSource();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
        backing = reinterpretOp.getSource();
        continue;
      }
      if (auto sv = dyn_cast<memref::SubViewOp>(def)) {
        backing = sv.getSource();
        continue;
      }
      break;
    }

    bool backingIsShared = false;
    if (auto *def = backing.getDefiningOp()) {
      backingIsShared = isa<memref::AllocOp>(def) || isa<memref::GetGlobalOp>(def);
    }

    if (backingIsShared) {
      std::string baseName = getName(backing).str().str();
      if (baseName.empty()) {
        emitError(op,
                  "pynq.data_transfer_instr requires operands backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global)");
        return;
      }
      sharedTempName = baseName + "_shared";
    } else {
      // If the source isn't backed by CMA shared memory, stage through a
      // temporary shared buffer (no pointer/state mutation).
      std::string normName =
          std::string("pynq_norm_sm") + std::to_string(sharedMemTempCounter++);
      indent();
      os << "PYNQ_SHARED_MEMORY " << normName << ";\n";
      indent();
      os << "PYNQ_allocatedSharedMemory(&" << normName << ", "
         << totalBytesForSize << ", 1);\n";
      sharedTempName = normName;
      stagingFromNormal = true;
      stagingFromNormalBytes = totalBytesForSize;
      stagingFromNormalPtr = subviewSourceName;
    }
  }
  
  // Try to get constant direction value
  auto directionValue = op.getDirection();
  
  // Check if direction is a constant
  if (auto constOp = directionValue.getDefiningOp<arith::ConstantOp>()) {
    auto directionAttr = constOp.getValue().dyn_cast<IntegerAttr>();
    if (directionAttr) {
      int64_t direction = directionAttr.getInt();
      
      indent();
      if (direction == 0) {
        if (stagingFromNormal) {
          os << "PYNQ_copyNormal2Share(&" << sharedTempName << ", (const void*)("
             << getCTypeName(subviewResultType.getElementType()) << "*)"
             << stagingFromNormalPtr << ", " << stagingFromNormalBytes
             << ");\n";
          indent();
        }
        os << "exec_data_load(&" << sharedTempName << ", ";
        emitValue(op.getBufferId());
        os << ", ";
        emitValue(op.getTileCount());
        os << ", ";
        emitValue(op.getTotalPkgNum());
        os << ");";
        emitInfoAndNewLine(op);
        return;
      }
      if (direction == 1) {
        os << "exec_data_store(&" << sharedTempName << ", ";
        emitValue(op.getBufferId());
        os << ", ";
        emitValue(op.getTileCount());
        os << ", ";
        emitValue(op.getTotalPkgNum());
        os << ");";
        emitInfoAndNewLine(op);
        if (stagingFromNormal) {
          indent();
          os << "pynq_sync();";
          emitInfoAndNewLine(op);
          indent();
            os << "PYNQ_copyShare2Normal((void*)("
             << getCTypeName(subviewResultType.getElementType()) << "*)"
             << stagingFromNormalPtr << ", " << stagingFromNormalBytes
              << "), &" << sharedTempName
             << ");";
          emitInfoAndNewLine(op);
          return;
        }
        if (unpackAfterStore) {
          auto elementType = subviewResultType.getElementType();
          auto rowStrides =
              computeRowMajorStrides(subviewResultType.getShape());
          unsigned loopId = subviewTempCounter++;

          indent();
          os << "pynq_sync();";
          emitInfoAndNewLine(op);

           // NOTE: sharedTempName comes from the staging buffer allocation above,
           // which already declared `<sharedTempName>_ptr` in this same scope.
          indent();
          os << getCTypeName(elementType) << "* " << sharedTempName
             << "_dst = (" << getCTypeName(elementType) << "*)"
             << subviewSourceName << ";\n";

          for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
            indent();
            os << "for (int sv" << loopId << "_" << i << " = 0; sv" << loopId
               << "_" << i << " < " << subviewResultType.getShape()[i]
               << "; ++sv" << loopId << "_" << i << ") {\n";
            addIndent();
          }

          indent();
          os << sharedTempName << "_dst[" << subviewOffset;
          for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
            os << " + sv" << loopId << "_" << i << " * " << subviewStrides[i];
          }
          os << "] = " << sharedTempName << "_ptr[";
          for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
            if (i != 0)
              os << " + ";
            os << "sv" << loopId << "_" << i << " * " << rowStrides[i];
          }
          os << "];\n";

          for (size_t i = 0; i < subviewResultType.getRank(); ++i) {
            reduceIndent();
            indent();
            os << "}\n";
          }
        }
        return;
      }
      if (direction == 2 || direction == 3) {
        // direction=2: host->matrix_fifo (target=0)
        // direction=3: host->vector_fifo (target=1)
        int64_t target = (direction == 2) ? 0 : 1;
        if (stagingFromNormal) {
          os << "PYNQ_copyNormal2Share(&" << sharedTempName << ", (const void*)("
             << getCTypeName(subviewResultType.getElementType()) << "*)"
             << stagingFromNormalPtr << ", " << stagingFromNormalBytes
             << ");\n";
          indent();
        }
        os << "exec_data_to_fifo(&" << sharedTempName << ", ";
        emitValue(op.getTotalPkgNum());
        os << ", " << target << ");";
        emitInfoAndNewLine(op);
        return;
      }

      // Unknown constant direction: fall back to store to keep C compilable.
      os << "exec_data_store(&" << sharedTempName << ", ";
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
  
  // If direction is not constant, emit runtime conditional chain.
  indent();
  os << "if (";
  emitValue(directionValue);
  os << " == 0) {\n";
  addIndent();
  if (stagingFromNormal) {
    indent();
    os << "PYNQ_copyNormal2Share(&" << sharedTempName << ", (const void*)("
       << getCTypeName(subviewResultType.getElementType()) << "*)"
       << stagingFromNormalPtr << ", " << stagingFromNormalBytes << ");\n";
  }
  indent();
  os << "exec_data_load(&" << sharedTempName << ", ";
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getTotalPkgNum());
  os << ");\n";
  reduceIndent();
  indent();
  os << "} else if (";
  emitValue(directionValue);
  os << " == 1) {\n";
  addIndent();
  indent();
  os << "exec_data_store(&" << sharedTempName << ", ";
  emitValue(op.getBufferId());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getTotalPkgNum());
  os << ");\n";
  if (stagingFromNormal) {
    indent();
    os << "pynq_sync();\n";
    indent();
     os << "PYNQ_copyShare2Normal((void*)("
       << getCTypeName(subviewResultType.getElementType()) << "*)"
       << stagingFromNormalPtr << ", " << stagingFromNormalBytes << "), &" << sharedTempName << ");\n";
  }
  reduceIndent();
  indent();
  os << "} else if (";
  emitValue(directionValue);
  os << " == 2) {\n";
  addIndent();
  if (stagingFromNormal) {
    indent();
    os << "PYNQ_copyNormal2Share(&" << sharedTempName << ", (const void*)("
       << getCTypeName(subviewResultType.getElementType()) << "*)"
       << stagingFromNormalPtr << ", " << stagingFromNormalBytes << ");\n";
  }
  indent();
  os << "exec_data_to_fifo(&" << sharedTempName << ", ";
  emitValue(op.getTotalPkgNum());
  os << ", 0);\n";
  reduceIndent();
  indent();
  os << "} else if (";
  emitValue(directionValue);
  os << " == 3) {\n";
  addIndent();
  if (stagingFromNormal) {
    indent();
    os << "PYNQ_copyNormal2Share(&" << sharedTempName << ", (const void*)("
       << getCTypeName(subviewResultType.getElementType()) << "*)"
       << stagingFromNormalPtr << ", " << stagingFromNormalBytes << ");\n";
  }
  indent();
  os << "exec_data_to_fifo(&" << sharedTempName << ", ";
  emitValue(op.getTotalPkgNum());
  os << ", 1);\n";
  reduceIndent();
  indent();
  os << "} else {\n";
  addIndent();
  indent();
  os << "// Unknown direction; defaulting to store\n";
  indent();
  os << "exec_data_store(&" << sharedTempName << ", ";
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

void PYNQCEmitter::emitBufferId(Value bufferLike) {
  if (auto bufferType = llvm::dyn_cast<pynq::BufferType>(bufferLike.getType())) {
    // Prefer emitting the concrete ID (after allocation).
    if (bufferType.isAllocated()) {
      os << bufferType.getBufferId();
      return;
    }
    // Virtual buffers don't have a concrete ID yet; emit something stable.
    // Later lowering/codegen passes should have eliminated this before final C.
    os << 0;
    return;
  }
  // Fallback: treat as a normal SSA value (e.g., for instr ops using i32 IDs).
  emitValue(bufferLike);
}

void PYNQCEmitter::emitCopy(pynq::CopyOp op) {
  // pynq.copy is an abstract data movement op. In the current pipeline it is
  // expected to be lowered/expanded into pynq.data_transfer_instr and/or scalar
  // loops before final codegen.
  indent();
  os << "// pynq.copy: ";
  emitValue(op.getSrc());
  os << " -> ";
  emitValue(op.getDst());
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitView(pynq::ViewOp op) {
  // pynq.view is a logical metadata op that affects how the runtime interprets
  // subsequent uses of the same underlying buffer.
  indent();
  os << "// pynq.view: ";
  emitValue(op.getInput());
  os << " -> ";
  emitValue(op.getOutput());
    os << " (row_splits=" << op.getRowSplits()
      << ", col_splits=" << op.getColSplits() << ")";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitBufferAlloc(pynq::BufferAllocOp op) {
  // Buffer allocation is typically resolved at compile time
  // Emit as a comment or placeholder for tracking
  indent();
  os << "// pynq.buffer_alloc: buffer_id=" << op.getBufferId()
     << ", capacity=" << op.getCapacityBytes() << " bytes";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSync(pynq::SyncOp op) {
  indent();
  os << "pynq_sync();";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSetMagic(pynq::SetMagicOp op) {
  auto magicAttr = op->getAttrOfType<IntegerAttr>("magic");
  int64_t magic = magicAttr ? magicAttr.getInt() : 0;
  indent();
  // os << "// pynq.setMagic: magic=" << magic;
  os << "write_magic_num_reg(" << magic << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitGELU(pynq::GELUOp op) {
  indent();
  os << "exec_gelu(";
  
  // GELU operation parameters
  emitBufferId(op.getBuffer());
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
  emitBufferId(op.getBuffer());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ", ";
  emitBufferId(op.getExtraBuffer());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSoftmax(pynq::SoftmaxOp op) {
  indent();
  os << "exec_softmax(";
  
  // Softmax operation parameters
  emitBufferId(op.getBuffer());
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
  emitBufferId(op.getBuffer());
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
  emitBufferId(op.getBuffer());
  os << ", ";
  emitValue(op.getTileCount());
  os << ", ";
  emitValue(op.getReduceK());
  os << ", ";
  emitValue(op.getOp());
  os << ", ";
  emitBufferId(op.getExtraBuffer());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitVectorInstr(pynq::VectorInstrOp op) {
  // Low-level instruction op mirroring gen_vector_instr() encoding.
  // We emit via the runtime helper which generates the instruction and writes
  // it to the vector target register.
  indent();
  os << "exec_vector(";
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

void PYNQCEmitter::emitActivationLayoutTranspose(
    pynq::ActivationLayoutTransposeOp op) {
  Value output = op.getOutput();
  Value input = op.getInput();

  auto outType = dyn_cast<MemRefType>(output.getType());
  auto inType = dyn_cast<MemRefType>(input.getType());
  if (!outType || !inType) {
    emitError(op, "pynq.activation_layout_transpose requires memref types");
    return;
  }
  if (!outType.hasStaticShape() || !inType.hasStaticShape()) {
    emitError(op, "pynq.activation_layout_transpose requires static shapes");
    return;
  }

  // Runtime transpose helpers we can target.
  Type elemType = inType.getElementType();
  enum class RuntimeTransposeKind { I8, F32 };
  std::optional<RuntimeTransposeKind> runtimeKind;

  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    if (intTy.getWidth() == 8)
      runtimeKind = RuntimeTransposeKind::I8;
  } else if (elemType.isF32()) {
    runtimeKind = RuntimeTransposeKind::F32;
  }

  if (!runtimeKind) {
    emitError(op,
              "pynq.activation_layout_transpose only supports i8 or f32 memref currently");
    return;
  }

  // Determine rows/cols for the 2D transpose.
  // For rank-3 (B,L,D) <-> (B,D,L), we currently require B == 1 so the
  // underlying storage matches a simple 2D matrix.
  int64_t rows = -1;
  int64_t cols = -1;
  ArrayRef<int64_t> inShape = inType.getShape();

  if (inType.getRank() == 2) {
    rows = inShape[0];
    cols = inShape[1];
  } else if (inType.getRank() == 3) {
    if (inShape[0] != 1) {
      emitError(op,
                "pynq.activation_layout_transpose only supports batch=1 currently");
      return;
    }
    rows = inShape[1];
    cols = inShape[2];
  } else {
    emitError(op,
              "pynq.activation_layout_transpose supports only rank-2 or rank-3 memref");
    return;
  }

  auto stripViews = [](Value v) -> Value {
    while (auto *def = v.getDefiningOp()) {
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        v = castOp.getSource();
        continue;
      }
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(def)) {
        v = subviewOp.getSource();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
        v = reinterpretOp.getSource();
        continue;
      }
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
        v = reshapeOp.getSource();
        continue;
      }
      break;
    }
    return v;
  };

  auto resolveSharedName = [&](Value v) -> std::optional<std::string> {
    Value base = stripViews(v);
    if (!base.getDefiningOp<memref::AllocOp>() &&
        !base.getDefiningOp<memref::GetGlobalOp>()) {
      return std::nullopt;
    }
    std::string baseName = getName(base).str().str();
    if (baseName.empty())
      return std::nullopt;
    return baseName + "_shared";
  };

  auto inShared = resolveSharedName(input);
  auto outShared = resolveSharedName(output);
  bool inputIsArg = input.isa<BlockArgument>();
  bool outputIsArg = output.isa<BlockArgument>();
  if (!outShared && !outputIsArg) {
    emitError(op,
              "pynq.activation_layout_transpose requires output backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global) or a function argument");
    return;
  }
  if (!inShared && !inputIsArg) {
    emitError(op,
              "pynq.activation_layout_transpose requires input backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global) or a function argument");
    return;
  }

  indent();
  if (*runtimeKind == RuntimeTransposeKind::I8) {
    if (inShared && outShared) {
      os << "PYNQ_transpose_int8(&" << *outShared << ", &" << *inShared
         << ", " << rows << ", " << cols << ");";
    } else if (!inShared && outShared) {
      auto inName = getName(input).str().str();
      if (inName.empty()) {
        emitError(op, "pynq.activation_layout_transpose requires a named input argument");
        return;
      }
        os << "PYNQ_transpose_int8_from_host(&" << *outShared << ", " << inName
         << ", " << rows << ", " << cols << ");";
    } else if (inShared && !outShared) {
      auto outName = getName(output).str().str();
      if (outName.empty()) {
        emitError(op, "pynq.activation_layout_transpose requires a named output argument");
        return;
      }
        os << "PYNQ_transpose_int8_to_host(" << outName << ", &" << *inShared
         << ", " << rows << ", " << cols << ");";
    } else {
      auto inName = getName(input).str().str();
      auto outName = getName(output).str().str();
      if (inName.empty() || outName.empty()) {
        emitError(op, "pynq.activation_layout_transpose requires named input/output arguments");
        return;
      }
        os << "PYNQ_transpose_int8_host_to_host(" << outName << ", " << inName
         << ", " << rows << ", " << cols << ");";
    }
  } else {
    if (inShared && outShared) {
      os << "PYNQ_transpose_fp32(&" << *outShared << ", &" << *inShared
         << ", " << rows << ", " << cols << ");";
    } else if (!inShared && outShared) {
      auto inName = getName(input).str().str();
      if (inName.empty()) {
        emitError(op, "pynq.activation_layout_transpose requires a named input argument");
        return;
      }
        os << "PYNQ_transpose_fp32_from_host(&" << *outShared << ", " << inName
         << ", " << rows << ", " << cols << ");";
    } else if (inShared && !outShared) {
      auto outName = getName(output).str().str();
      if (outName.empty()) {
        emitError(op, "pynq.activation_layout_transpose requires a named output argument");
        return;
      }
        os << "PYNQ_transpose_fp32_to_host(" << outName << ", &" << *inShared
         << ", " << rows << ", " << cols << ");";
    } else {
      auto inName = getName(input).str().str();
      auto outName = getName(output).str().str();
      if (inName.empty() || outName.empty()) {
        emitError(op, "pynq.activation_layout_transpose requires named input/output arguments");
        return;
      }
        os << "PYNQ_transpose_fp32_host_to_host(" << outName << ", " << inName
         << ", " << rows << ", " << cols << ");";
    }
  }
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitActivationDynamicPad(pynq::ActivationDynamicPadOp op) {
  Value output = op.getOutput();
  Value input = op.getInput();
  const bool toPadded = op.getToPadded();

  auto outType = dyn_cast<MemRefType>(output.getType());
  auto inType = dyn_cast<MemRefType>(input.getType());
  if (!outType || !inType) {
    emitError(op, "pynq.activation_dynamic_pad requires memref types");
    return;
  }
  if (!outType.hasStaticShape() || !inType.hasStaticShape()) {
    emitError(op, "pynq.activation_dynamic_pad requires static shapes");
    return;
  }
  if (outType.getRank() < 1 || inType.getRank() < 1) {
    emitError(op, "pynq.activation_dynamic_pad requires rank >= 1");
    return;
  }

  auto inElemI = dyn_cast<IntegerType>(inType.getElementType());
  auto outElemI = dyn_cast<IntegerType>(outType.getElementType());
  if (!inElemI || !outElemI || inElemI.getWidth() != 8 || outElemI.getWidth() != 8) {
    emitError(op, "pynq.activation_dynamic_pad currently supports only i8 memref");
    return;
  }

  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  const int64_t inRank = inType.getRank();
  const int64_t outRank = outType.getRank();

  // Validate row-major contiguous layout when static strides are available.
  auto requireRowMajorContiguous = [&](MemRefType t, StringRef which) -> bool {
    SmallVector<int64_t, 4> strides;
    int64_t offset = 0;
    if (!getStaticStridesAndOffset(t, strides, offset)) {
      emitError(op, ("pynq.activation_dynamic_pad requires static strides for " + which).str());
      return false;
    }
    if (offset != 0 || !isRowMajorContiguous(t.getShape(), strides)) {
      emitError(op, ("pynq.activation_dynamic_pad requires row-major contiguous layout for " + which).str());
      return false;
    }
    return true;
  };
  if (!requireRowMajorContiguous(inType, "input") ||
      !requireRowMajorContiguous(outType, "output"))
    return;

  // Compute runtime arguments depending on semantic mode.
  int64_t rows = 0;
  int64_t colsIn = 0;
  int64_t colsOut = 0;
  int64_t outer = 0;
  int64_t rowsIn2D = 0;
  int64_t colsIn2D = 0;
  int64_t rowsOut2D = 0;
  int64_t colsOut2D = 0;
  if (toPadded) {
    // Last-dimension pad/truncate: require same rank and matching leading dims.
    if (outRank != inRank) {
      emitError(op, "pynq.activation_dynamic_pad(to_padded=true) requires input/output ranks to match");
      return;
    }
    const int64_t rank = inRank;
    for (int64_t i = 0; i < rank - 1; ++i) {
      if (inShape[i] != outShape[i]) {
        emitError(op, "pynq.activation_dynamic_pad(to_padded=true) requires leading dims to match (pad only along last dim)");
        return;
      }
    }
    rows = 1;
    for (int64_t i = 0; i < rank - 1; ++i)
      rows *= inShape[i];
    colsIn = inShape[rank - 1];
    colsOut = outShape[rank - 1];
  } else {
    // Fused subview(offset=0)+copy (clone-style) over the last two dims.
    // We ignore the leading dims by flattening them into `outer`.
    if (outRank != inRank) {
      emitError(op, "pynq.activation_dynamic_pad(to_padded=false) requires input/output ranks to match");
      return;
    }
    if (inRank < 2) {
      emitError(op, "pynq.activation_dynamic_pad(to_padded=false) requires rank >= 2 to copy a 2D slice (last two dims)");
      return;
    }
    for (int64_t i = 0; i < inRank - 2; ++i) {
      if (inShape[i] != outShape[i]) {
        emitError(op, "pynq.activation_dynamic_pad(to_padded=false) requires leading dims (batch-like) to match");
        return;
      }
    }
    outer = 1;
    for (int64_t i = 0; i < inRank - 2; ++i)
      outer *= inShape[i];

    rowsIn2D = inShape[inRank - 2];
    colsIn2D = inShape[inRank - 1];
    rowsOut2D = outShape[outRank - 2];
    colsOut2D = outShape[outRank - 1];

    if (rowsOut2D > rowsIn2D || colsOut2D > colsIn2D) {
      emitError(op, "pynq.activation_dynamic_pad(to_padded=false) requires output[last2] <= input[last2] (top-left subview0 copy)");
      return;
    }
  }

  auto stripViews = [](Value v) -> Value {
    while (auto *def = v.getDefiningOp()) {
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        v = castOp.getSource();
        continue;
      }
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(def)) {
        v = subviewOp.getSource();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
        v = reinterpretOp.getSource();
        continue;
      }
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
        v = reshapeOp.getSource();
        continue;
      }
      break;
    }
    return v;
  };

  auto resolveSharedName = [&](Value v) -> std::optional<std::string> {
    Value base = stripViews(v);
    if (!base.getDefiningOp<memref::AllocOp>() &&
        !base.getDefiningOp<memref::GetGlobalOp>()) {
      return std::nullopt;
    }
    std::string baseName = getName(base).str().str();
    if (baseName.empty())
      return std::nullopt;
    return baseName + "_shared";
  };

  auto inShared = resolveSharedName(input);
  auto outShared = resolveSharedName(output);
  if (!inShared || !outShared) {
    emitError(op,
              "pynq.activation_dynamic_pad requires operands backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global)");
    return;
  }

  indent();
  if (toPadded) {
    os << "PYNQ_activation_dynamic_pad_int8(&" << *outShared << ", &" << *inShared
       << ", " << rows << ", " << colsIn << ", " << colsOut << ");";
  } else {
     os << "PYNQ_activation_subview0_copy_int8(&" << *outShared << ", &" << *inShared
       << ", " << outer << ", " << rowsIn2D << ", " << colsIn2D << ", " << rowsOut2D << ", " << colsOut2D << ");";
  }
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitQuant(pynq::QuantOp op) {
  Value output = op.getOutput();
  Value input = op.getInput();
  Value scale = op.getScale();
  Value zero = op.getZero();

  auto outType = dyn_cast<MemRefType>(output.getType());
  auto inType = dyn_cast<MemRefType>(input.getType());
  auto scaleType = dyn_cast<MemRefType>(scale.getType());
  if (!outType || !inType || !scaleType) {
    emitError(op, "pynq.quant requires memref types for output/input/scale");
    return;
  }
  if (!outType.hasStaticShape() || !inType.hasStaticShape()) {
    emitError(op, "pynq.quant requires static shapes for output/input");
    return;
  }
  if (outType.getRank() != inType.getRank() || outType.getShape() != inType.getShape()) {
    emitError(op, "pynq.quant requires output and input shapes to match");
    return;
  }

  if (!inType.getElementType().isF32()) {
    emitError(op, "pynq.quant currently supports only f32 input");
    return;
  }
  auto outElemI = dyn_cast<IntegerType>(outType.getElementType());
  if (!outElemI || outElemI.getWidth() != 8) {
    emitError(op, "pynq.quant currently supports only i8 output");
    return;
  }

  auto scaleElemI = dyn_cast<IntegerType>(scaleType.getElementType());
  if (!scaleElemI || scaleElemI.getWidth() != 32) {
    emitError(op, "pynq.quant requires packed scale to be i32 memref");
    return;
  }

  // Derive rows/cols by flattening leading dims into rows.
  const int64_t rank = inType.getRank();
  if (rank < 1) {
    emitError(op, "pynq.quant requires rank >= 1");
    return;
  }
  ArrayRef<int64_t> shape = inType.getShape();
  int64_t rows = 1;
  for (int64_t i = 0; i < rank - 1; ++i)
    rows *= shape[i];
  int64_t cols = shape[rank - 1];

  auto stripViews = [](Value v) -> Value {
    while (auto *def = v.getDefiningOp()) {
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        v = castOp.getSource();
        continue;
      }
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(def)) {
        v = subviewOp.getSource();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
        v = reinterpretOp.getSource();
        continue;
      }
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
        v = reshapeOp.getSource();
        continue;
      }
      break;
    }
    return v;
  };

  auto resolveSharedName = [&](Value v) -> std::optional<std::string> {
    Value base = stripViews(v);
    if (!base.getDefiningOp<memref::AllocOp>() &&
        !base.getDefiningOp<memref::GetGlobalOp>()) {
      return std::nullopt;
    }
    std::string baseName = getName(base).str().str();
    if (baseName.empty())
      return std::nullopt;
    return baseName + "_shared";
  };

  auto outShared = resolveSharedName(output);
  auto inShared = resolveSharedName(input);
  auto scaleShared = resolveSharedName(scale);
  std::optional<std::string> zeroShared;
  if (zero)
    zeroShared = resolveSharedName(zero);

  if (!outShared || !inShared || !scaleShared) {
    emitError(op,
              "pynq.quant requires output/input/scale backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global)");
    return;
  }

  auto cmbAttr = op->getAttrOfType<IntegerAttr>("quant_cmb");
  int64_t cmbRaw = cmbAttr ? cmbAttr.getInt() : 0;
  if (cmbRaw < 0 || cmbRaw > 255) {
    emitError(op, "pynq.quant quant_cmb must be representable in i8");
    return;
  }

  uint8_t quantCmb = static_cast<uint8_t>(cmbRaw) & 0x7u;
  bool isAsymmetric = (quantCmb & 0x1u) != 0;
  bool is1D = (quantCmb & 0x2u) != 0;
  bool alongCol = (quantCmb & 0x4u) != 0;

  const char *callee = nullptr;
  bool needsZero = isAsymmetric;

  if (!is1D) {
    callee = isAsymmetric ? "PYNQ_quant_fp32_to_int8_asym_0d"
                          : "PYNQ_quant_fp32_to_int8_sym_0d";
  } else if (alongCol) {
    callee = isAsymmetric ? "PYNQ_quant_fp32_to_int8_asym_col"
                          : "PYNQ_quant_fp32_to_int8_sym_col";
  } else {
    callee = isAsymmetric ? "PYNQ_quant_fp32_to_int8_asym_row"
                          : "PYNQ_quant_fp32_to_int8_sym_row";
  }

  if (needsZero) {
    if (!zero) {
      emitError(op, "pynq.quant asymmetric mode requires zero operand");
      return;
    }
    if (!zeroShared) {
      emitError(op, "pynq.quant zero must be backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global)");
      return;
    }
    auto zeroType = dyn_cast<MemRefType>(zero.getType());
    auto zeroElemI = zeroType ? dyn_cast<IntegerType>(zeroType.getElementType())
                              : IntegerType();
    if (!zeroType || !zeroElemI || zeroElemI.getWidth() != 32) {
      emitError(op, "pynq.quant currently requires zero-point memref element type i32 for asymmetric modes");
      return;
    }
  }

  indent();
  os << callee << "(&" << *outShared << ", &" << *inShared << ", &"
     << *scaleShared;
  if (needsZero)
    os << ", &" << *zeroShared;
  os << ", " << rows << ", " << cols << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitDequant(pynq::DequantOp op) {
  Value output = op.getOutput();
  Value input = op.getInput();
  Value scale = op.getScale();
  Value zero = op.getZero();

  auto outType = dyn_cast<MemRefType>(output.getType());
  auto inType = dyn_cast<MemRefType>(input.getType());
  auto scaleType = dyn_cast<MemRefType>(scale.getType());
  if (!outType || !inType || !scaleType) {
    emitError(op, "pynq.dequant requires memref types for output/input/scale");
    return;
  }
  if (!outType.hasStaticShape() || !inType.hasStaticShape()) {
    emitError(op, "pynq.dequant requires static shapes for output/input");
    return;
  }
  if (outType.getRank() != inType.getRank() || outType.getShape() != inType.getShape()) {
    emitError(op, "pynq.dequant requires output and input shapes to match");
    return;
  }

  if (!outType.getElementType().isF32()) {
    emitError(op, "pynq.dequant currently supports only f32 output");
    return;
  }
  auto inElemI = dyn_cast<IntegerType>(inType.getElementType());
  if (!inElemI || inElemI.getWidth() != 8) {
    emitError(op, "pynq.dequant currently supports only i8 input");
    return;
  }

  auto scaleElemI = dyn_cast<IntegerType>(scaleType.getElementType());
  if (!scaleElemI || scaleElemI.getWidth() != 32) {
    emitError(op, "pynq.dequant requires packed scale to be i32 memref");
    return;
  }

  const int64_t rank = inType.getRank();
  if (rank < 1) {
    emitError(op, "pynq.dequant requires rank >= 1");
    return;
  }
  ArrayRef<int64_t> shape = inType.getShape();
  int64_t rows = 1;
  for (int64_t i = 0; i < rank - 1; ++i)
    rows *= shape[i];
  int64_t cols = shape[rank - 1];

  auto stripViews = [](Value v) -> Value {
    while (auto *def = v.getDefiningOp()) {
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        v = castOp.getSource();
        continue;
      }
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(def)) {
        v = subviewOp.getSource();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
        v = reinterpretOp.getSource();
        continue;
      }
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
        v = reshapeOp.getSource();
        continue;
      }
      break;
    }
    return v;
  };

  auto resolveSharedName = [&](Value v) -> std::optional<std::string> {
    Value base = stripViews(v);
    if (!base.getDefiningOp<memref::AllocOp>() &&
        !base.getDefiningOp<memref::GetGlobalOp>()) {
      return std::nullopt;
    }
    std::string baseName = getName(base).str().str();
    if (baseName.empty())
      return std::nullopt;
    return baseName + "_shared";
  };

  auto outShared = resolveSharedName(output);
  auto inShared = resolveSharedName(input);
  auto scaleShared = resolveSharedName(scale);
  std::optional<std::string> zeroShared;
  if (zero)
    zeroShared = resolveSharedName(zero);

  if (!outShared || !inShared || !scaleShared) {
    emitError(op,
              "pynq.dequant requires output/input/scale backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global)");
    return;
  }

  auto cmbAttr = op->getAttrOfType<IntegerAttr>("quant_cmb");
  int64_t cmbRaw = cmbAttr ? cmbAttr.getInt() : 0;
  if (cmbRaw < 0 || cmbRaw > 255) {
    emitError(op, "pynq.dequant quant_cmb must be representable in i8");
    return;
  }

  uint8_t quantCmb = static_cast<uint8_t>(cmbRaw) & 0x7u;
  bool isAsymmetric = (quantCmb & 0x1u) != 0;
  bool is1D = (quantCmb & 0x2u) != 0;
  bool alongCol = (quantCmb & 0x4u) != 0;

  const char *callee = nullptr;
  bool needsZero = isAsymmetric;

  if (!is1D) {
    callee = isAsymmetric ? "PYNQ_dequant_int8_to_fp32_asym_0d"
                          : "PYNQ_dequant_int8_to_fp32_sym_0d";
  } else if (alongCol) {
    callee = isAsymmetric ? "PYNQ_dequant_int8_to_fp32_asym_col"
                          : "PYNQ_dequant_int8_to_fp32_sym_col";
  } else {
    callee = isAsymmetric ? "PYNQ_dequant_int8_to_fp32_asym_row"
                          : "PYNQ_dequant_int8_to_fp32_sym_row";
  }

  if (needsZero) {
    if (!zero) {
      emitError(op, "pynq.dequant asymmetric mode requires zero operand");
      return;
    }
    if (!zeroShared) {
      emitError(op, "pynq.dequant zero must be backed by PYNQ_SHARED_MEMORY (memref.alloc or memref.get_global)");
      return;
    }
    auto zeroType = dyn_cast<MemRefType>(zero.getType());
    auto zeroElemI = zeroType ? dyn_cast<IntegerType>(zeroType.getElementType())
                              : IntegerType();
    if (!zeroType || !zeroElemI || zeroElemI.getWidth() != 32) {
      emitError(op, "pynq.dequant currently requires zero-point memref element type i32 for asymmetric modes");
      return;
    }
  }

  indent();
  os << callee << "(&" << *outShared << ", &" << *inShared << ", &"
     << *scaleShared;
  if (needsZero)
    os << ", &" << *zeroShared;
  os << ", " << rows << ", " << cols << ");";
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
  os << "__attribute__((unused)) ";
  emitMemrefVarDecl(memrefType, varName);
  os << " = (";
  emitMemrefPtrCastType(memrefType);
  os << ")" << sharedMemName << ".pointer;\n";
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

void PYNQCEmitter::emitMemrefCopy(memref::CopyOp op) {
  Value src = op.getSource();
  Value dst = op.getTarget();

  auto srcType = src.getType().dyn_cast<MemRefType>();
  auto dstType = dst.getType().dyn_cast<MemRefType>();
  if (!srcType || !dstType) {
    emitError(op, "memref.copy requires memref types");
    return;
  }
  if (!srcType.hasStaticShape() || !dstType.hasStaticShape()) {
    emitError(op, "memref.copy with dynamic shapes not supported");
    return;
  }
  if (srcType.getElementType() != dstType.getElementType() ||
      srcType.getNumElements() != dstType.getNumElements()) {
    emitError(op, "memref.copy requires matching element type and element count");
    return;
  }

  int64_t elemBits = srcType.getElementType().getIntOrFloatBitWidth();
  int64_t elemBytes = (elemBits + 7) / 8;
  int64_t totalBytes = srcType.getNumElements() * elemBytes;

  auto stripViews = [](Value v) -> Value {
    while (auto *def = v.getDefiningOp()) {
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        v = castOp.getSource();
        continue;
      }
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(def)) {
        v = subviewOp.getSource();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
        v = reinterpretOp.getSource();
        continue;
      }
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
        v = reshapeOp.getSource();
        continue;
      }
      break;
    }
    return v;
  };

  Value srcBase = stripViews(src);
  Value dstBase = stripViews(dst);

  auto isSharedBase = [](Value v) -> bool {
    return v.getDefiningOp<memref::AllocOp>() || v.getDefiningOp<memref::GetGlobalOp>();
  };

  bool srcShared = isSharedBase(srcBase);
  bool dstShared = isSharedBase(dstBase);

  auto buildDataPtrExpr = [&](Value v) -> std::optional<std::string> {
    // Build an element-pointer expression for `v`, accumulating any static
    // view offsets (subview/reinterpret_cast) along the way.
    int64_t totalOffset = 0;
    Value base = v;
    while (auto *def = base.getDefiningOp()) {
      if (auto castOp = dyn_cast<memref::CastOp>(def)) {
        base = castOp.getSource();
        continue;
      }
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(def)) {
        base = reshapeOp.getSource();
        continue;
      }
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(def)) {
        auto subType = subviewOp.getType().cast<MemRefType>();
        SmallVector<int64_t, 4> subStrides;
        int64_t subOffset = 0;
        if (!getStaticStridesAndOffset(subType, subStrides, subOffset))
          return std::nullopt;
        totalOffset += subOffset;
        base = subviewOp.getSource();
        continue;
      }
      if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(def)) {
        auto reType = reinterpretOp.getResult().getType().dyn_cast<MemRefType>();
        if (!reType)
          return std::nullopt;

        SmallVector<int64_t, 4> reStrides;
        int64_t reOffsetFromType = 0;
        if (!getStaticStridesAndOffset(reType, reStrides, reOffsetFromType))
          return std::nullopt;

        // Keep in sync with emitReinterpretCast: prefer explicit op offset;
        // fall back to the type offset when present.
        int64_t opOffset = 0;
        auto mixedOffsets = reinterpretOp.getMixedOffsets();
        if (mixedOffsets.size() != 1)
          return std::nullopt;
        if (Attribute attr = mixedOffsets[0].dyn_cast<Attribute>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(attr))
            opOffset = intAttr.getInt();
          else
            return std::nullopt;
        } else if (Value offVal = mixedOffsets[0].dyn_cast<Value>()) {
          if (auto constOp = offVal.getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
              opOffset = intAttr.getInt();
            else
              return std::nullopt;
          } else {
            return std::nullopt;
          }
        } else {
          return std::nullopt;
        }

        int64_t effectiveOffset = opOffset;
        if (effectiveOffset == 0)
          effectiveOffset = reOffsetFromType;
        totalOffset += effectiveOffset;
        base = reinterpretOp.getSource();
        continue;
      }
      break;
    }

    auto baseName = getName(base);
    if (baseName.empty())
      return std::nullopt;

    auto elemCType = getCTypeName(srcType.getElementType());
    std::string expr = "(" + elemCType.str().str() + "*)(" + baseName.str().str() + ")";
    if (totalOffset != 0)
      expr += " + " + std::to_string(totalOffset);
    return expr;
  };

  auto srcPtrExpr = buildDataPtrExpr(src);
  auto dstPtrExpr = buildDataPtrExpr(dst);
  if (!srcPtrExpr || !dstPtrExpr) {
    emitError(op, "memref.copy operands must be lowered to a materializable base pointer");
    return;
  }

  indent();
  if (srcShared && !dstShared) {
    auto srcBaseName = getName(srcBase);
    if (srcBaseName.empty()) {
      emitError(op, "memref.copy: shared source has no base name");
      return;
    }
    os << "PYNQ_copyShare2Normal((void*)" << *dstPtrExpr << ", &" << srcBaseName
       << "_shared, " << totalBytes << ");";
  } else if (!srcShared && dstShared) {
    auto dstBaseName = getName(dstBase);
    if (dstBaseName.empty()) {
      emitError(op, "memref.copy: shared target has no base name");
      return;
    }
    os << "PYNQ_copyNormal2Share(&" << dstBaseName
       << "_shared, (const void*)" << *srcPtrExpr << ", " << totalBytes << ");";
  } else {
    // shared->shared or normal->normal
    os << "memcpy((void*)(" << *dstPtrExpr << "), (const void*)(" << *srcPtrExpr << ")"
       << ", " << totalBytes << ");";
  }
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitCast(memref::CastOp op) {
  if (isDeclared(op.getResult()))
    return;

  auto resultType = op.getResult().getType().dyn_cast<MemRefType>();
  auto sourceType = op.getSource().getType().dyn_cast<MemRefType>();
  if (!resultType || !sourceType) {
    indent();
    emitValue(op.getResult());
    os << " = ";
    emitValue(op.getSource());
    os << ";";
    emitInfoAndNewLine(op);
    return;
  }

  auto srcName = getName(op.getSource());
  if (srcName.empty()) {
    emitError(op, "memref.cast source must be declared before cast");
    return;
  }

  indent();
  auto varName = addName(op.getResult(), false);
  os << "__attribute__((unused)) ";
  emitMemrefVarDecl(resultType, varName);
  os << " = (";
  emitMemrefPtrCastType(resultType);
  os << ")" << srcName << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitReinterpretCast(memref::ReinterpretCastOp op) {
  if (isDeclared(op.getResult()))
    return;

  auto resultType = op.getResult().getType().dyn_cast<MemRefType>();
  auto sourceType = op.getSource().getType().dyn_cast<MemRefType>();
  if (!resultType || !sourceType) {
    emitError(op, "memref.reinterpret_cast requires memref types");
    return;
  }

  if (resultType.getElementType() != sourceType.getElementType()) {
    emitError(op, "memref.reinterpret_cast with differing element type not supported");
    return;
  }

  SmallVector<int64_t, 4> resStrides;
  int64_t resOffsetFromType = 0;
  if (!getStaticStridesAndOffset(resultType, resStrides, resOffsetFromType)) {
    emitError(op, "dynamic reinterpret_cast stride/offset not supported");
    return;
  }

  if (!isRowMajorContiguous(resultType.getShape(), resStrides)) {
    emitError(op, "non-contiguous memref.reinterpret_cast not supported");
    return;
  }

  // The op carries an explicit offset operand/result; for our pointer-only
  // lowering we require it to be a compile-time constant. (Most pipelines
  // materialize it as `offset: [0]`.)
  int64_t opOffset = 0;
  auto mixedOffsets = op.getMixedOffsets();
  if (mixedOffsets.size() != 1) {
    emitError(op, "memref.reinterpret_cast with non-scalar base offset not supported");
    return;
  }
  if (Attribute attr = mixedOffsets[0].dyn_cast<Attribute>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      opOffset = intAttr.getInt();
    } else {
      emitError(op, "memref.reinterpret_cast offset attribute must be integer");
      return;
    }
  } else if (Value val = mixedOffsets[0].dyn_cast<Value>()) {
    if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
        opOffset = intAttr.getInt();
      else {
        emitError(op, "memref.reinterpret_cast offset must be an integer constant");
        return;
      }
    } else {
      emitError(op, "memref.reinterpret_cast offset must be constant");
      return;
    }
  } else {
    emitError(op, "memref.reinterpret_cast offset must be constant");
    return;
  }

  // Prefer the explicit op offset; fall back to the type offset when present.
  // (In well-formed IR these should agree.)
  int64_t effectiveOffset = opOffset;
  if (effectiveOffset == 0)
    effectiveOffset = resOffsetFromType;

  auto srcName = getName(op.getSource());
  if (srcName.empty()) {
    emitError(op, "memref.reinterpret_cast source must be declared before reinterpret_cast");
    return;
  }

  // We only model memrefs as a base pointer in generated C. This is correct for
  // the common case where reinterpret_cast is used as a view/reshape for DMA.
  // If the resulting view is later indexed as a multi-dimensional array in C,
  // the surrounding pipeline should lower those accesses before reaching here.
  indent();
  os << "__attribute__((unused)) "
     << getCTypeName(resultType.getElementType()) << "* "
     << addName(op.getResult(), false) << " = ("
     << getCTypeName(resultType.getElementType()) << "*)" << srcName;
  if (effectiveOffset != 0)
    os << " + " << effectiveOffset;
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitReshape(memref::ReshapeOp op) {
  if (isDeclared(op.getResult()))
    return;

  auto resultType = op.getResult().getType().dyn_cast<MemRefType>();
  auto sourceType = op.getSource().getType().dyn_cast<MemRefType>();
  if (!resultType || !sourceType) {
    emitError(op, "memref.reshape requires memref types");
    return;
  }
  if (resultType.getElementType() != sourceType.getElementType()) {
    emitError(op, "memref.reshape with differing element type not supported");
    return;
  }

  auto srcName = getName(op.getSource());
  if (srcName.empty()) {
    emitError(op, "memref.reshape source must be declared before reshape");
    return;
  }

  // Lower reshape as a pure view: same base pointer with a cast.
  // The shape operand is ignored here; later IR is expected to have explicit
  // indexing lowered such that only the base pointer matters.
  indent();
  auto varName = addName(op.getResult(), false);
  os << "__attribute__((unused)) ";
  emitMemrefVarDecl(resultType, varName);
  os << " = (";
  emitMemrefPtrCastType(resultType);
  os << ")" << srcName << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSubview(memref::SubViewOp op) {
  if (isDeclared(op.getResult()))
    return;

  auto resultType = op.getType().cast<MemRefType>();
  auto sourceType = op.getSource().getType().cast<MemRefType>();

  SmallVector<int64_t, 4> resStrides;
  int64_t resOffset = 0;
  if (!getStaticStridesAndOffset(resultType, resStrides, resOffset)) {
    emitError(op, "dynamic subview stride/offset not supported");
    return;
  }

  SmallVector<int64_t, 4> srcStrides;
  int64_t srcOffset = 0;
  if (!getStaticStridesAndOffset(sourceType, srcStrides, srcOffset)) {
    emitError(op, "dynamic source stride/offset not supported for subview");
    return;
  }

  bool identity = (resultType.getShape() == sourceType.getShape()) &&
                  (resOffset == srcOffset) && (resStrides == srcStrides);

  bool contiguous = isRowMajorContiguous(resultType.getShape(), resStrides);

  if (!identity && !contiguous) {
    bool ok = true;
    for (auto *user : op.getResult().getUsers()) {
      auto transfer = dyn_cast<pynq::DataTransferInstrOp>(user);
      if (!transfer) {
        ok = false;
        break;
      }
      int64_t dir = -1;
      if (auto constOp = transfer.getDirection().getDefiningOp<arith::ConstantOp>()) {
        if (auto directionAttr = constOp.getValue().dyn_cast<IntegerAttr>())
          dir = directionAttr.getInt();
      }
      if (dir != 0 && dir != 1 && dir != 2 && dir != 3) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      op.emitWarning("non-contiguous subview requires DMA pack or is unsupported");
      return;
    }

    indent();
    os << "// non-contiguous subview handled by DMA packing";
    emitInfoAndNewLine(op);
    return;
  }

  auto srcName = getName(op.getSource());
  if (srcName.empty()) {
    emitError(op, "subview source must be declared before subview");
    return;
  }

  indent();
  os << "__attribute__((unused)) "
     << getCTypeName(resultType.getElementType()) << "* "
     << addName(op.getResult(), false) << " = ("
     << getCTypeName(resultType.getElementType()) << "*)" << srcName;
  if (!identity && resOffset != 0)
    os << " + " << resOffset;
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitGetGlobal(memref::GetGlobalOp op) {
  auto memrefType = op.getType().cast<ShapedType>();
  if (!memrefType.hasStaticShape()) {
    emitError(op, "dynamic shape not supported for memref.get_global");
    return;
  }

  // Calculate total size in bytes
  int64_t totalElements = 1;
  for (auto dim : memrefType.getShape())
    totalElements *= dim;
  auto elementType = memrefType.getElementType();
  unsigned elementSizeBits = elementType.getIntOrFloatBitWidth();
  unsigned elementSizeBytes = (elementSizeBits + 7) / 8;
  int64_t totalBytes = totalElements * elementSizeBytes;
  int64_t alignedTotalBytes =
      computeAlignedByteSize(totalElements, elementSizeBytes);

  auto varName = addName(op.getResult(), false);
  auto sharedMemName = varName + "_shared";

  indent();
  os << "PYNQ_SHARED_MEMORY " << sharedMemName << ";";
  emitInfoAndNewLine(op);

  if (embedGlobalData) {
    indent();
    os << "PYNQ_allocatedSharedMemory(&" << sharedMemName << ", "
       << totalBytes << ", 1);\n";
    indent();
    os << "PYNQ_copyNormal2Share(&" << sharedMemName << ", "
       << op.getName() << ", " << totalBytes << ");\n";
  } else {
    indent();
    os << "PYNQ_allocatedSharedMemory(&" << sharedMemName << ", "
       << alignedTotalBytes << ", 1);\n";
    indent();
    os << "PYNQ_loadBin(&" << sharedMemName << ", \"" << globalBinDir
       << "/" << op.getName() << ".bin\", " << alignedTotalBytes << ");\n";
  }

  indent();
  auto memrefTy = op.getType().cast<MemRefType>();
  // os << "c ";
  emitMemrefVarDecl(memrefTy, varName);
  os << " = (";
  emitMemrefPtrCastType(memrefTy);
  os << ")" << sharedMemName << ".pointer;";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitGlobal(memref::GlobalOp op) {
  auto init_val = op.getInitialValue();
  if (!init_val.has_value())
    return;

  auto attr = init_val.value();
  auto denseAttr = attr.dyn_cast<DenseElementsAttr>();
  if (!denseAttr)
    return;

  auto arrayType = op.getType().cast<ShapedType>();
  auto type = arrayType.getElementType();

  if (embedGlobalData) {
    indent();
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
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";
      } else if (type.isF64()) {
        auto value = element.cast<FloatAttr>().getValue().convertToDouble();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";
      } else if (type.isInteger(1))
        os << element.cast<BoolAttr>().getValue();
      else if (type.isIntOrIndex()) {
        auto intType = type.dyn_cast<IntegerType>();
        os << element.cast<IntegerAttr>().getValue();
        if (intType && intType.getWidth() > 64)
          os << "LL";
      } else {
        emitError(op, "array has unsupported element type.");
      }

      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
    return;
  }

  if (llvm::sys::fs::create_directories(globalParentDir + "/" + globalBinDir)) {
    emitError(op, "failed to create global binary output directory");
    return;
  }

  SmallString<256> binPath(globalParentDir + "/" + globalBinDir);
  llvm::sys::path::append(binPath, op.getSymName().str() + ".bin");
  std::string errorMsg;
  if (!writeDenseElementsToBinary(denseAttr, type, binPath, errorMsg)) {
    emitError(op, errorMsg);
    return;
  }

  indent();
  os << "// exported global: " << op.getSymName() << " -> " << binPath;
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitCall(func::CallOp op) {
  // Only direct calls are supported in C emission.
  auto calleeAttr = op.getCalleeAttr();
  if (!calleeAttr) {
    emitError(op, "indirect call not supported");
    return;
  }

  if (op.getNumResults() > 1) {
    emitError(op, "multiple return values not supported");
    return;
  }

  indent();
  if (op.getNumResults() == 1) {
    Value result = op.getResult(0);
    if (auto memrefType = result.getType().dyn_cast<MemRefType>()) {
      auto varName = addName(result, false);
      emitMemrefVarDecl(memrefType, varName);
      os << " = (";
      emitMemrefPtrCastType(memrefType);
      os << ")";
    } else {
      emitValue(result);
      os << " = ";
    }
  }

  os << calleeAttr.getValue() << "(";
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (i)
      os << ", ";
    emitValue(op.getOperand(i));
  }
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitReturn(func::ReturnOp op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  unsigned numFuncResults = 0;
  if (parentFunc)
    numFuncResults = parentFunc.getFunctionType().getNumResults();

  // Keep C compilable even if IR is inconsistent.
  if (numFuncResults == 0) {
    if (op.getNumOperands() == 0)
      return;
    indent();
    os << "return;";
    emitInfoAndNewLine(op);
    return;
  }

  if (numFuncResults > 1) {
    emitError(op, "multiple return values not supported");
    return;
  }

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
  auto iterArgs = op.getInitArgs();
  auto regionArgs = op.getRegionIterArgs();

  if (!iterArgs.empty()) {
    for (auto [initVal, regionArg] : llvm::zip(iterArgs, regionArgs)) {
      indent();
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

  // Loop initialization: declare and initialize the induction variable.
  // Note: emitValue() prints the type when the value is first declared.
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
  emitBlock(*op.getBody());

  // Handle scf.yield: update loop-carried variables
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
// PYNQCEmitter Implementation - arith Operations
//===----------------------------------------------------------------------===//

void PYNQCEmitter::emitAddI(arith::AddIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " + ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSubI(arith::SubIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " - ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitMulI(arith::MulIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " * ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitDivSI(arith::DivSIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " / ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitDivUI(arith::DivUIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto lhsTy = op.getLhs().getType().dyn_cast<IntegerType>();
  auto rhsTy = op.getRhs().getType().dyn_cast<IntegerType>();
  if (lhsTy && rhsTy && lhsTy.getWidth() == rhsTy.getWidth()) {
    auto uTy = getUnsignedCIntTypeName(lhsTy);
    os << "((" << uTy << ")";
    emitValue(op.getLhs());
    os << ") / ((" << uTy << ")";
    emitValue(op.getRhs());
    os << ")";
  } else {
    emitValue(op.getLhs());
    os << " / ";
    emitValue(op.getRhs());
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitRemSI(arith::RemSIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " % ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitRemUI(arith::RemUIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto lhsTy = op.getLhs().getType().dyn_cast<IntegerType>();
  auto rhsTy = op.getRhs().getType().dyn_cast<IntegerType>();
  if (lhsTy && rhsTy && lhsTy.getWidth() == rhsTy.getWidth()) {
    auto uTy = getUnsignedCIntTypeName(lhsTy);
    os << "((" << uTy << ")";
    emitValue(op.getLhs());
    os << ") % ((" << uTy << ")";
    emitValue(op.getRhs());
    os << ")";
  } else {
    emitValue(op.getLhs());
    os << " % ";
    emitValue(op.getRhs());
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitAndI(arith::AndIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " & ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitOrI(arith::OrIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " | ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitXorI(arith::XOrIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " ^ ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitShLI(arith::ShLIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " << ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitShRSI(arith::ShRSIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " >> ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitShRUI(arith::ShRUIOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto lhsTy = op.getLhs().getType().dyn_cast<IntegerType>();
  if (lhsTy) {
    auto uTy = getUnsignedCIntTypeName(lhsTy);
    os << "((" << uTy << ")";
    emitValue(op.getLhs());
    os << ") >> ";
    emitValue(op.getRhs());
  } else {
    emitValue(op.getLhs());
    os << " >> ";
    emitValue(op.getRhs());
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitAddF(arith::AddFOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " + ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSubF(arith::SubFOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " - ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitMulF(arith::MulFOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " * ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitDivF(arith::DivFOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getLhs());
  os << " / ";
  emitValue(op.getRhs());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitNegF(arith::NegFOp op) {
  indent();
  emitValue(op.getResult());
  os << " = -";
  emitValue(op.getOperand());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitCmpI(arith::CmpIOp op) {
  auto pred = op.getPredicate();
  auto lhsTy = op.getLhs().getType().dyn_cast<IntegerType>();

  auto isUnsignedPred = [&]() {
    return pred == arith::CmpIPredicate::ult || pred == arith::CmpIPredicate::ule ||
           pred == arith::CmpIPredicate::ugt || pred == arith::CmpIPredicate::uge;
  };

  auto emitOperandMaybeUnsigned = [&](Value v) {
    if (lhsTy && isUnsignedPred()) {
      auto uTy = getUnsignedCIntTypeName(lhsTy);
      os << "((" << uTy << ")";
      emitValue(v);
      os << ")";
      return;
    }
    emitValue(v);
  };

  const char *cmp = nullptr;
  switch (pred) {
  case arith::CmpIPredicate::eq:
    cmp = "==";
    break;
  case arith::CmpIPredicate::ne:
    cmp = "!=";
    break;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    cmp = "<";
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    cmp = "<=";
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    cmp = ">";
    break;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    cmp = ">=";
    break;
  default:
    cmp = "==";
    break;
  }

  indent();
  emitValue(op.getResult());
  os << " = (";
  emitOperandMaybeUnsigned(op.getLhs());
  os << " " << cmp << " ";
  emitOperandMaybeUnsigned(op.getRhs());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitCmpF(arith::CmpFOp op) {
  auto pred = op.getPredicate();

  auto emitIsNaN = [&](Value v) {
    os << "(";
    emitValue(v);
    os << " != ";
    emitValue(v);
    os << ")";
  };

  indent();
  emitValue(op.getResult());
  os << " = (";
  switch (pred) {
  case arith::CmpFPredicate::OEQ:
    emitValue(op.getLhs());
    os << " == ";
    emitValue(op.getRhs());
    break;
  case arith::CmpFPredicate::ONE:
    emitValue(op.getLhs());
    os << " != ";
    emitValue(op.getRhs());
    break;
  case arith::CmpFPredicate::OLT:
    emitValue(op.getLhs());
    os << " < ";
    emitValue(op.getRhs());
    break;
  case arith::CmpFPredicate::OLE:
    emitValue(op.getLhs());
    os << " <= ";
    emitValue(op.getRhs());
    break;
  case arith::CmpFPredicate::OGT:
    emitValue(op.getLhs());
    os << " > ";
    emitValue(op.getRhs());
    break;
  case arith::CmpFPredicate::OGE:
    emitValue(op.getLhs());
    os << " >= ";
    emitValue(op.getRhs());
    break;
  case arith::CmpFPredicate::ORD:
    os << "!(";
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    os << ")";
    break;
  case arith::CmpFPredicate::UNO:
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    break;
  case arith::CmpFPredicate::UEQ:
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    os << " || (";
    emitValue(op.getLhs());
    os << " == ";
    emitValue(op.getRhs());
    os << ")";
    break;
  case arith::CmpFPredicate::UNE:
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    os << " || (";
    emitValue(op.getLhs());
    os << " != ";
    emitValue(op.getRhs());
    os << ")";
    break;
  case arith::CmpFPredicate::ULT:
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    os << " || (";
    emitValue(op.getLhs());
    os << " < ";
    emitValue(op.getRhs());
    os << ")";
    break;
  case arith::CmpFPredicate::ULE:
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    os << " || (";
    emitValue(op.getLhs());
    os << " <= ";
    emitValue(op.getRhs());
    os << ")";
    break;
  case arith::CmpFPredicate::UGT:
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    os << " || (";
    emitValue(op.getLhs());
    os << " > ";
    emitValue(op.getRhs());
    os << ")";
    break;
  case arith::CmpFPredicate::UGE:
    emitIsNaN(op.getLhs());
    os << " || ";
    emitIsNaN(op.getRhs());
    os << " || (";
    emitValue(op.getLhs());
    os << " >= ";
    emitValue(op.getRhs());
    os << ")";
    break;
  default:
    emitValue(op.getLhs());
    os << " == ";
    emitValue(op.getRhs());
    break;
  }
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSelect(arith::SelectOp op) {
  indent();
  emitValue(op.getResult());
  os << " = (";
  emitValue(op.getCondition());
  os << " ? ";
  emitValue(op.getTrueValue());
  os << " : ";
  emitValue(op.getFalseValue());
  os << ");";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitExtSI(arith::ExtSIOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitExtUI(arith::ExtUIOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitTruncI(arith::TruncIOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitSIToFP(arith::SIToFPOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitUIToFP(arith::UIToFPOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitFPToSI(arith::FPToSIOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitFPToUI(arith::FPToUIOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitExtF(arith::ExtFOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitTruncF(arith::TruncFOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitIndexCast(arith::IndexCastOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
}

void PYNQCEmitter::emitIndexCastUI(arith::IndexCastUIOp op) {
  indent();
  emitValue(op.getOut());
  os << " = (" << getCTypeName(op.getOut().getType()) << ")";
  emitValue(op.getIn());
  os << ";";
  emitInfoAndNewLine(op);
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
  
  // Never encode pointer-ness into the stored name. The Allo name table
  // prefixes a '*' when isPtr=true, which breaks later reuse as an identifier.
  if (name.empty())
    os << addName(val, /*isPtr=*/false);
  else
    os << addName(val, /*isPtr=*/false, name);
  
  for (unsigned i = 0; i < rank; ++i)
    os << "[iv" << i << "]";
}

void PYNQCEmitter::emitMemrefVarDecl(MemRefType memrefType, StringRef name) {
  Type elementType = memrefType.getElementType();
  os << getCTypeName(elementType) << " ";

  if (!memrefType.hasStaticShape() || memrefType.getRank() <= 1) {
    os << "* " << name;
    return;
  }

  os << "(*" << name << ")";
  ArrayRef<int64_t> shape = memrefType.getShape();
  for (int64_t i = 1, e = memrefType.getRank(); i < e; ++i)
    os << "[" << shape[i] << "]";
}

void PYNQCEmitter::emitMemrefPtrCastType(MemRefType memrefType) {
  Type elementType = memrefType.getElementType();

  if (!memrefType.hasStaticShape() || memrefType.getRank() <= 1) {
    os << getCTypeName(elementType) << "*";
    return;
  }

  os << getCTypeName(elementType) << " (*)";
  ArrayRef<int64_t> shape = memrefType.getShape();
  for (int64_t i = 1, e = memrefType.getRank(); i < e; ++i)
    os << "[" << shape[i] << "]";
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
  
  // Emit function signature.
  // Use the MLIR function type for the return type so that the emitted
  // definition matches emitted calls and emitted return statements.
  auto funcType = func.getFunctionType();
  if (funcType.getNumResults() > 1) {
    emitError(func, "multiple return values not supported");
    return;
  }

  // Special-case memref returns: for rank>1 static shapes, C/C++ requires the
  // function name to be inside the pointer-to-array declarator:
  //   T (*f(args))[d1]...[dN]
  // not:
  //   T (*)[d1]...[dN] f(args)
  bool needsMemrefReturnSuffix = false;
  MemRefType memrefRetType;
  if (funcType.getNumResults() == 0) {
    os << "void " << func.getName() << "(\n";
  } else {
    Type retType = funcType.getResult(0);
    if (auto mr = retType.dyn_cast<MemRefType>()) {
      memrefRetType = mr;
      Type elementType = mr.getElementType();
      if (mr.hasStaticShape() && mr.getRank() > 1) {
        os << getCTypeName(elementType) << " (*" << func.getName() << "(\n";
        needsMemrefReturnSuffix = true;
      } else {
        os << getCTypeName(elementType) << "* " << func.getName() << "(\n";
      }
    } else {
      os << getCTypeName(retType) << " " << func.getName() << "(\n";
    }
  }
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
  
  reduceIndent();
  if (needsMemrefReturnSuffix) {
    // Close the parameter list and the pointer declarator, then emit the array
    // suffix for dimensions 1..N.
    os << "\n))";
    ArrayRef<int64_t> shape = memrefRetType.getShape();
    for (int64_t i = 1, e = memrefRetType.getRank(); i < e; ++i)
      os << "[" << shape[i] << "]";
    os << " {\n";
  } else {
    os << "\n) {\n";
  }
  
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
#include <string.h>
#include "instr.h"  // Hardware instruction generation functions
#include "pynq_runtime.h"  // PYNQ runtime API (exec_* helpers)
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
  if (auto embedAttr = module->getAttrOfType<BoolAttr>("pynq.embed_global_data"))
    embedGlobalData = embedAttr.getValue();
  if (auto binDirAttr = module->getAttrOfType<StringAttr>("pynq.global_bin_dir"))
    globalBinDir = binDirAttr.getValue().str();
  if (auto projectDirAttr = module->getAttrOfType<StringAttr>("allo.target_path"))
    globalParentDir = projectDirAttr.getValue().str();

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
