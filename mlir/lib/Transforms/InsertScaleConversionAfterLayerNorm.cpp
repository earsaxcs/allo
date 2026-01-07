/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Insert Scale Conversion After LayerNorm Pass
 * 
 * This pass inserts vivado.scale_conversion operations after vivado.int_layernorm
 * operations to handle scale mode mismatches between LayerNorm output and
 * the next consumer operation.
 * 
 * **Motivation:**
 * LayerNorm typically outputs per-channel quantization (scale per dimension),
 * but subsequent operations (e.g., QMatMul, QLinear) may expect per-token
 * quantization. This pass automatically inserts the necessary scale conversion.
 * 
 * **Transformation:**
 * Before:
 *   vivado.int_layernorm(...) -> %ln_out with per-channel scale
 *   vivado.qmatmul(%output, %ln_out, ...) expects per-token scale
 * 
 * After:
 *   vivado.int_layernorm(...) -> %ln_out with per-channel scale
 *   %converted = memref.alloc : same type as %ln_out
 *   vivado.scale_conversion(%converted, %ln_out, 
 *                           %ln_oscl, %next_op_iscl,
 *                           input_scale_mode = 1,  // per-channel
 *                           output_scale_mode = 2) // per-token
 *   vivado.qmatmul(%output, %converted, ...)
 * 
 * **Scale Mode Detection:**
 * - Per-tensor (0): scale is scalar (no dimensions)
 * - Per-channel (1): scale has dimension matching D (embedding dimension)
 * - Per-token (2): scale has dimension matching L (sequence length)
 * 
 * NOTE: Currently, output_scale_mode is hardcoded based on simple heuristics.
 * Future work should query the next operation's expected scale mode from
 * operation attributes or interface methods.
 */

#include "PassDetail.h"
#include "allo/Transforms/Passes.h"
#include "allo/Dialect/VivadoOps.h"
#include "allo/Dialect/PYNQConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "insert-scale-conversion-after-layernorm"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

namespace vivado_ops = mlir::allo::vivado;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Determine if LayerNorm output scale is per-channel or per-tensor
/// Returns: ScaleMode::kPerChannel or ScaleMode::kPerTensor
/// LayerNorm output can be:
/// - per-tensor: single scale for all elements
/// - per-channel: scale per D (embedding) dimension
static uint8_t determineLayerNormOutputScaleMode(Value scale, int64_t channelDim) {
  // Check if scale is a scalar (per-tensor)
  if (auto scalarType = scale.getType().dyn_cast<IntegerType>()) {
    return ScaleMode::kPerTensor;
  }
  
  // Check if scale is a memref/tensor
  if (auto shapedType = scale.getType().dyn_cast<ShapedType>()) {
    ArrayRef<int64_t> shape = shapedType.getShape();
    
    // Scalar memref/tensor (rank-0 or single element)
    if (shape.empty() || (shape.size() == 1 && shape[0] == 1)) {
      return ScaleMode::kPerTensor;
    }
    
    // Check for per-channel (matches D dimension)
    if (shape.size() == 1 && shape[0] == channelDim) {
      return ScaleMode::kPerChannel;
    }
    
    // Multi-dimensional scale with last dim matching channelDim
    if (shape.size() > 1 && shape.back() == channelDim) {
      return ScaleMode::kPerChannel;
    }
  }
  
  // Default to per-tensor if cannot determine
  return ScaleMode::kPerTensor;
}

/// Determine if next operation expects per-token or per-tensor scale
/// Returns: ScaleMode::kPerToken or ScaleMode::kPerTensor
/// Next operations after LayerNorm typically expect:
/// - per-tensor: single scale for all elements
/// - per-token: scale per L (sequence length) dimension
static uint8_t determineNextOpExpectedScaleMode(Value scale, int64_t tokenDim) {
  // Check if scale is a scalar (per-tensor)
  if (auto scalarType = scale.getType().dyn_cast<IntegerType>()) {
    return ScaleMode::kPerTensor;
  }
  
  // Check if scale is a memref/tensor
  if (auto shapedType = scale.getType().dyn_cast<ShapedType>()) {
    ArrayRef<int64_t> shape = shapedType.getShape();
    
    // Scalar memref/tensor (rank-0 or single element)
    if (shape.empty() || (shape.size() == 1 && shape[0] == 1)) {
      return ScaleMode::kPerTensor;
    }
    
    // Check for per-token (matches L dimension)
    if (shape.size() == 1 && shape[0] == tokenDim) {
      return ScaleMode::kPerToken;
    }

    // NOTE: We currently do NOT support per-token scales that vary across batch
    // (e.g., [B, L] or [B, ..., L]). All batches are assumed to share one set
    // of per-token scales, represented as a rank-1 shape [L].
  }
  
  // Default to per-tensor if cannot determine
  return ScaleMode::kPerTensor;
}

/// Get input scale from the next user operation
/// Returns null if no suitable user found or scale cannot be extracted
static Value getNextOpInputScale(Value output) {
  // Iterate through users of the LayerNorm output
  for (Operation *user : output.getUsers()) {
    // Check common quantized ops that have input scale
    if (auto qmatmulOp = dyn_cast<vivado_ops::QMatMulOp>(user)) {
      // Assuming output is lhs (first input)
      if (qmatmulOp.getLhs() == output) {
        return qmatmulOp.getXScale();
      }
      // If output is rhs (second input)
      if (qmatmulOp.getRhs() == output) {
        return qmatmulOp.getYScale();
      }
    } else if (auto qlinearOp = dyn_cast<vivado_ops::QLinearOp>(user)) {
      if (qlinearOp.getInput() == output) {
        return qlinearOp.getIscl();
      }
    } else if (auto qaddOp = dyn_cast<vivado_ops::QAddOp>(user)) {
      if (qaddOp.getLhs() == output) {
        return qaddOp.getXScale();
      }
      if (qaddOp.getRhs() == output) {
        return qaddOp.getYScale();
      }
    } else if (auto qmatmulIsqrtdOp = dyn_cast<vivado_ops::QMatMulIsqrtDOp>(user)) {
      if (qmatmulIsqrtdOp.getLhs() == output) {
        return qmatmulIsqrtdOp.getXScale();
      }
      if (qmatmulIsqrtdOp.getRhs() == output) {
        return qmatmulIsqrtdOp.getYScale();
      }
    }
  }
  
  return Value(); // No suitable user found
}

struct UserScaleAndZero {
  Value inputScale;
  Value inputZero;
};

/// Get the input scale/zero for a specific user op and operand position.
/// Returns {Value(), Value()} if the user is not a supported consumer.
static UserScaleAndZero getUserInputScaleAndZero(Operation *user, Value producerOutput) {
  if (auto qmatmulOp = dyn_cast<vivado_ops::QMatMulOp>(user)) {
    if (qmatmulOp.getLhs() == producerOutput) {
      return {qmatmulOp.getXScale(), qmatmulOp.getXZero()};
    }
    if (qmatmulOp.getRhs() == producerOutput) {
      return {qmatmulOp.getYScale(), qmatmulOp.getYZero()};
    }
  } else if (auto qlinearOp = dyn_cast<vivado_ops::QLinearOp>(user)) {
    if (qlinearOp.getInput() == producerOutput) {
      return {qlinearOp.getIscl(), qlinearOp.getInputZero()};
    }
  } else if (auto qaddOp = dyn_cast<vivado_ops::QAddOp>(user)) {
    if (qaddOp.getLhs() == producerOutput) {
      return {qaddOp.getXScale(), qaddOp.getXZero()};
    }
    if (qaddOp.getRhs() == producerOutput) {
      return {qaddOp.getYScale(), qaddOp.getYZero()};
    }
  } else if (auto qmatmulIsqrtdOp = dyn_cast<vivado_ops::QMatMulIsqrtDOp>(user)) {
    if (qmatmulIsqrtdOp.getLhs() == producerOutput) {
      return {qmatmulIsqrtdOp.getXScale(), qmatmulIsqrtdOp.getXZero()};
    }
    if (qmatmulIsqrtdOp.getRhs() == producerOutput) {
      return {qmatmulIsqrtdOp.getYScale(), qmatmulIsqrtdOp.getYZero()};
    }
  }

  return {Value(), Value()};
}

//===----------------------------------------------------------------------===//
// InsertScaleConversionAfterLayerNormPass Implementation
//===----------------------------------------------------------------------===//

class InsertScaleConversionAfterLayerNormPass 
    : public PassWrapper<InsertScaleConversionAfterLayerNormPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertScaleConversionAfterLayerNormPass)

  InsertScaleConversionAfterLayerNormPass() = default;
  InsertScaleConversionAfterLayerNormPass(const InsertScaleConversionAfterLayerNormPass &) {}

  StringRef getArgument() const final { 
    return "insert-scale-conversion-after-layernorm"; 
  }
  
  StringRef getDescription() const final {
    return "Insert vivado.scale_conversion after vivado.int_layernorm to handle scale mode mismatches";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vivado::VivadoDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    
    if (!applyInsertScaleConversionAfterLayerNorm(module, context)) {
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

bool applyInsertScaleConversionAfterLayerNorm(ModuleOp &module, MLIRContext *context) {
  // Collect IntLayerNorm ops
  SmallVector<vivado_ops::IntLayerNormOp, 8> layerNormOps;
  
  module.walk([&](vivado_ops::IntLayerNormOp op) {
    LLVM_DEBUG(llvm::dbgs() << "Found IntLayerNorm at " << op.getLoc() << "\n");
    layerNormOps.push_back(op);
  });
  
  if (layerNormOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No IntLayerNorm ops found\n");
    return true;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << layerNormOps.size() 
                          << " IntLayerNorm ops to process\n");
  
  // Process each IntLayerNorm op
  for (vivado_ops::IntLayerNormOp lnOp : layerNormOps) {
    Value lnOutput = lnOp.getOutput();
    Value lnOscl = lnOp.getOscl();
    Value lnOutputZero = lnOp.getOutputZero();
    
    // Get output shape to determine dimensions
    // NOTE: This pass assumes bufferization has already happened and LN outputs are memrefs.
    // Tensor outputs are not supported here.
    auto outputType = lnOutput.getType().cast<MemRefType>();
    ArrayRef<int64_t> shape = outputType.getShape();

    // Expect activation shapes like (B, L, D) or (B, D, L).
    // If rank is not 3, we skip rather than guessing semantics.
    if (shape.size() < 3) {
      LLVM_DEBUG(llvm::dbgs() << "LN output rank < 3, skipping\n");
      continue;
    }
    
    // Determine the output layout using the new contract:
    // - transpose_mode gates whether transpose semantics are enabled.
    // - is_transposed only has meaning when transpose_mode=true.
    bool transposeMode = lnOp.getTransposeMode();
    bool layoutTransposed = transposeMode && lnOp.getIsTransposed();
    int64_t channelDim = layoutTransposed ? shape[1] : shape[2]; // D dimension
    int64_t tokenDim = layoutTransposed ? shape[2] : shape[1];   // L dimension
    
    // Determine input scale mode (LayerNorm output scale)
    // LayerNorm output: per-channel (scale per D) or per-tensor
    uint8_t inputScaleMode = determineLayerNormOutputScaleMode(lnOscl, channelDim);
    
    LLVM_DEBUG(llvm::dbgs() << "LayerNorm output scale mode: " 
                            << static_cast<int>(inputScaleMode) 
                            << " (" << static_cast<int>(ScaleMode::kPerTensor) << "=per-tensor, " 
                            << static_cast<int>(ScaleMode::kPerChannel) << "=per-channel)\n");
    

    // Insert scale conversion per-use. Although we detect from IntLayerNorm,
    // the actual conversion is placed right before each consumer use so that
    // multiple consumers with different expected scale modes can be handled.
    SmallVector<OpOperand *, 8> uses;
    uses.reserve(std::distance(lnOutput.use_begin(), lnOutput.use_end()));
    for (OpOperand &use : lnOutput.getUses()) {
      uses.push_back(&use);
    }

    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();

      UserScaleAndZero next = getUserInputScaleAndZero(user, lnOutput);
      if (!next.inputScale) {
        continue;
      }

      // Determine output scale mode (next operation's expected scale)
      // Next operation after LayerNorm: per-token (scale per L) or per-tensor
      // NOTE: Currently only checks scale dimensions. Future work could add
      // operation-specific attributes to explicitly declare expected scale mode.
      uint8_t outputScaleMode = determineNextOpExpectedScaleMode(next.inputScale, tokenDim);

      LLVM_DEBUG(llvm::dbgs() << "Next operation expected scale mode: "
                              << static_cast<int>(outputScaleMode)
                              << " (" << static_cast<int>(ScaleMode::kPerTensor) << "=per-tensor, "
                              << static_cast<int>(ScaleMode::kPerToken) << "=per-token)\n");

      // Skip if scale modes already match
      if (inputScaleMode == outputScaleMode) {
        continue;
      }

      OpBuilder builder(user);
      Location loc = lnOp.getLoc();

      // Allocate output buffer for converted result
      Value convertedOutput = builder.create<memref::AllocOp>(loc, outputType);

      // Create scale conversion op
      (void)builder.create<vivado_ops::ScaleConversionOp>(
          loc,
          convertedOutput,          // output
          lnOutput,                 // input
          lnOscl,                   // input_scale
          next.inputScale,          // output_scale
          lnOutputZero,             // input_zero
          next.inputZero,           // output_zero
          builder.getI8IntegerAttr(inputScaleMode),   // input_scale_mode
          builder.getI8IntegerAttr(outputScaleMode),  // output_scale_mode
          lnOp.getScalePackingAttr(),
          lnOp.getScaleCoeModeAttr(),
          lnOp.getTransposeModeAttr(),
          lnOp.getIsTransposedAttr());

      // Replace only this use.
      use->set(convertedOutput);

      LLVM_DEBUG(llvm::dbgs() << "Inserted scale_conversion before consumer\n");
    }
  }
  
  return true;
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createInsertScaleConversionAfterLayerNormPass() {
  return std::make_unique<InsertScaleConversionAfterLayerNormPass>();
}

} // namespace allo
} // namespace mlir
