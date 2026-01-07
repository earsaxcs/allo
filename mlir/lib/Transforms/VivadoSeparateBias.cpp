/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Vivado Separate Bias Pass
 * 
 * This pass separates the fused bias from vivado.qlinear operations into
 * separate vivado.qadd operations. This is useful when:
 * 1. The hardware doesn't support fused bias in matmul
 * 2. Bias needs to be applied with different quantization parameters
 * 3. Debugging or analysis requires explicit bias operations
 * 
 * **Transformation:**
 * Before:
 *   vivado.qlinear(%output, %input, %weight, ..., %bscl, ..., %bias)
 *     {fuse_bias = true}
 * 
 * After:
 *   %temp = memref.alloc : same type as output
 *   vivado.qlinear(%temp, %input, %weight, ..., /no bscl/, ..., /no bias/)
 *     {fuse_bias = false}
 *   vivado.qadd(%output, %temp, %bias_broadcasted, 
 *               %oscl, %bscl, %oscl, %oscl_inv, ...)
 * 
 * **Important Notes:**
 * 1. The QAdd's x_scale uses QLinear's output scale (oscl)
 * 2. The QAdd's y_scale uses QLinear's bias scale (bscl)
 * 3. Bias must be pre-broadcasted to match output shape, otherwise QAddOp
 *    verifier will emit an error
 */

#include "PassDetail.h"
#include "allo/Transforms/Passes.h"
#include "allo/Dialect/VivadoOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vivado-separate-bias"

using namespace mlir;
using namespace mlir::allo;

namespace vivado_ops = mlir::allo::vivado;

namespace {

static int64_t getBatchFromModule(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>("allo.batch"))
    return attr.getInt();
  return 1;
}

/// Ensure bias global has a leading batch dimension by repeating its initializer.
/// Only handles the common case where bias is loaded via memref.get_global.
///
/// We broadcast only along the leading batch dimension (B). Other dimensions are
/// assumed to already match the activation/output shape.
static LogicalResult ensureBiasGlobalBroadcastToBatch(ModuleOp module,
                                                     Value bias,
                                                     MemRefType outputType,
                                                     int64_t batch,
                                                     MLIRContext *context) {
  if (!bias)
    return success();
  if (batch <= 0)
    batch = 1;

  auto biasType = dyn_cast<MemRefType>(bias.getType());
  if (!biasType)
    return success();

  // If bias already matches output rank, assume it is already broadcasted.
  if (biasType.getRank() == outputType.getRank())
    return success();

  // Only handle the case where bias is missing exactly one leading dimension.
  if (biasType.getRank() != outputType.getRank() - 1)
    return success();

  auto getGlobal = bias.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return success();

  auto globalOp = module.lookupSymbol<memref::GlobalOp>(getGlobal.getName());
  if (!globalOp)
    return success();

  auto memrefType = globalOp.getType().dyn_cast<MemRefType>();
  if (!memrefType)
    return success();

  // Avoid rebroadcasting if this global was already updated.
  if (memrefType.getRank() == outputType.getRank())
    return success();

  SmallVector<int64_t, 4> newShape;
  newShape.reserve(memrefType.getRank() + 1);
  newShape.push_back(batch);
  for (int64_t d : memrefType.getShape())
    newShape.push_back(d);

  auto elementType = memrefType.getElementType();
  auto newType = MemRefType::get(newShape, elementType);

  // Update initializer by repeating the old dense data B times (if present).
  if (auto initialValue = globalOp.getInitialValue()) {
    if (auto denseAttr = initialValue->dyn_cast<DenseElementsAttr>()) {
      SmallVector<Attribute> oldValues;
      oldValues.reserve(denseAttr.getNumElements());
      for (Attribute v : denseAttr.getValues<Attribute>())
        oldValues.push_back(v);

      SmallVector<Attribute> newValues;
      newValues.reserve(oldValues.size() * static_cast<size_t>(batch));
      for (int64_t b = 0; b < batch; ++b)
        newValues.append(oldValues.begin(), oldValues.end());

      auto newTensorType = RankedTensorType::get(newShape, elementType);
      auto newDenseAttr = DenseElementsAttr::get(newTensorType, newValues);

      // Mutate global only after we have a compatible initializer.
      globalOp.setType(newType);
      globalOp.setInitialValueAttr(newDenseAttr);
    } else {
      // Initializer exists but isn't a dense constant; we don't know how to
      // safely repeat it here.
      return failure();
    }
  } else {
    // No initializer: still update the global's type so users see the batch dim.
    globalOp.setType(newType);
  }

  // Update all GetGlobalOp users to use the new type.
  auto symbolName = globalOp.getSymName();
  module.walk([&](memref::GetGlobalOp g) {
    if (g.getName() == symbolName)
      g.getResult().setType(newType);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// VivadoSeparateBiasPass Implementation
//===----------------------------------------------------------------------===//

class VivadoSeparateBiasPass 
    : public PassWrapper<VivadoSeparateBiasPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VivadoSeparateBiasPass)

  VivadoSeparateBiasPass() = default;
  VivadoSeparateBiasPass(const VivadoSeparateBiasPass &) {}

  StringRef getArgument() const final { 
    return "vivado-separate-bias"; 
  }
  
  StringRef getDescription() const final {
    return "Separate fused bias from vivado.qlinear into explicit vivado.qadd";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vivado::VivadoDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    
    if (!applyVivadoSeparateBias(module, context)) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Apply Function
//===----------------------------------------------------------------------===//

namespace mlir {
namespace allo {

bool applyVivadoSeparateBias(ModuleOp &module, MLIRContext *context) {
  const int64_t batch = getBatchFromModule(module);

  // Collect QLinear ops with fused bias
  SmallVector<vivado_ops::QLinearOp, 8> opsToSeparate;
  
  module.walk([&](vivado_ops::QLinearOp op) {
    // Check if this op has fused bias
    if (op.getFuseBias() && op.getBias()) {
      LLVM_DEBUG(llvm::dbgs() << "Found QLinear with fused bias at " 
                              << op.getLoc() << "\n");
      opsToSeparate.push_back(op);
    }
  });
  
  if (opsToSeparate.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No QLinear ops with fused bias found\n");
    return true;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << opsToSeparate.size() 
                          << " QLinear ops to separate bias\n");
  
  // Process each QLinear op
  for (vivado_ops::QLinearOp op : opsToSeparate) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    
    // Get original output and create temp buffer
    Value originalOutput = op.getOutput();
    auto outputType = originalOutput.getType().cast<MemRefType>();
    Value tempOutput = builder.create<memref::AllocOp>(loc, outputType);
    
    // Get bias and bias scale
    Value bias = op.getBias();
    Value bscl = op.getBscl();
    Value oscl = op.getOscl();
    Value osclInv = op.getOsclInv();
    Value outputZero = op.getOutputZero();

    // Ensure bias matches activation/output shape by broadcasting its global to batch.
    if (failed(ensureBiasGlobalBroadcastToBatch(module, bias, outputType, batch, context)))
      return false;
    
    // Create new QLinear without bias
    builder.create<vivado_ops::QLinearOp>(
        loc,
        tempOutput,           // output -> temp buffer
        op.getInput(),
        op.getWeight(),
        op.getFscl(),
        op.getIscl(),
        op.getOscl(),
        op.getOsclInv(),
        op.getWscl(),
        Value(),              // bscl = null (no bias scale)
        op.getInputZero(),
        op.getOutputZero(),
        Value(),              // bias = null (no bias)
        op.getTileMAttr(),
        op.getTileNAttr(),
        op.getTileKAttr(),
        op.getBufferStrategyAttr(),
        op.getBufferIdsAttr(),
        op.getBiasBufferIdAttr(),
        builder.getBoolAttr(false),  // fuse_bias = false
        op.getTransposeWeightAttr(),
        op.getHlsPragmasAttr(),
        op.getAccumulatorTypeAttr(),
        op.getRequantModeAttr(),
        op.getScalePackingAttr(),
        op.getScaleCoeModeAttr(),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr(),
        op.getLayerTypeAttr()
    );
    
    // Create QAdd for bias addition
    // x_scale = output scale (for QLinear output)
    // y_scale = bias scale (for bias)
    // o_scale = output scale (final output)
    builder.create<vivado_ops::QAddOp>(
        loc,
        originalOutput,       // output -> original output buffer
        tempOutput,           // lhs = QLinear output (temp)
        bias,                 // rhs = bias
        oscl,                 // x_scale = output scale
        bscl,                 // y_scale = bias scale
        oscl,                 // o_scale = output scale
        osclInv,              // o_scale_inv
        outputZero,           // x_zero
        outputZero,           // y_zero (assume same zero point for bias)
        outputZero,           // o_zero
        builder.getBoolAttr(false),  // fuse_into_producer
        builder.getBoolAttr(true),   // vectorize
        op.getScalePackingAttr(),
        op.getScaleCoeModeAttr(),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    // Erase the original operation
    op.erase();
  }
  
  return true;
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createVivadoSeparateBiasPass() {
  return std::make_unique<VivadoSeparateBiasPass>();
}

} // namespace allo
} // namespace mlir
