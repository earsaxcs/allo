/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Vivado QLinear K-Dimension Split Pass
 * 
 * This pass handles the case where the K dimension (reduction dimension) of
 * vivado.qlinear operations exceeds the hardware tile size limit (TileConfig::kDefaultTileK).
 * 
 * **Motivation:**
 * In FC2 layers of transformer models, the weight matrix dimensions can be quite large.
 * When K (the reduction/input dimension) exceeds the tile size limit:
 * - For M and N dimensions: can be handled later during Vivado->PYNQ lowering as independent tiles
 * - For K dimension: must be split early because partial results have data dependencies
 *   (each K-tile produces a partial sum that must be accumulated)
 * 
 * **Transformation:**
 * For a QLinear with K > TileK:
 * 
 * Before:
 *   vivado.qlinear(%output, %input, %weight, ...) : [B,L,K_large] x [OC,K_large] -> [B,L,OC]
 * 
 * After:
 *   // Split input along K dimension
 *   %input_0 = subview %input[..., 0:TileK]
 *   %input_1 = subview %input[..., TileK:2*TileK]
 *   ...
 *   
 *   // Split weight along K dimension (IC dimension for weight)
 *   %weight_0 = subview %weight[..., 0:TileK]
 *   %weight_1 = subview %weight[..., TileK:2*TileK]
 *   ...
 *   
 *   // Create partial QLinear operations
 *   %partial_0 = alloc buffer for partial result
 *   vivado.qlinear(%partial_0, %input_0, %weight_0, ...)
 *   
 *   %partial_1 = alloc buffer for partial result
 *   vivado.qlinear(%partial_1, %input_1, %weight_1, ...)
 *   ...
 *   
 *   // Accumulate partial results using QAdd
 *   %accum_0 = partial_0
 *   %accum_1 = qadd(%accum_0, %partial_1, o_scale, o_scale, o_scale, o_scale_inv)
 *   ...
 *   copy %accum_final -> %output
 * 
 * **Key Design Decisions:**
 * 1. The QAdd operations use the original QLinear's o_scale (packed scale, not inv) for both inputs
 *    since all partial results share the same quantization scale.
 * 2. Only K dimension is handled here; M and N dimensions are handled during PYNQ lowering
 *    because they don't have inter-tile data dependencies.
 * 3. The is_transposed attribute determines which dimension is K:
 *    - is_transposed=false: input (B,L,D), K=D (last dim of input, second dim of weight)
 *    - is_transposed=true:  input (B,D,L), K=L (second dim of input, first dim of weight)
 */

 #include "PassDetail.h"
#include "allo/Transforms/Passes.h"
#include "allo/Dialect/PYNQConfig.h"
#include "allo/Dialect/VivadoOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vivado-qlinear-k-split"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

// Use vivado_ops namespace to avoid ambiguity
namespace vivado_ops = mlir::allo::vivado;

//===----------------------------------------------------------------------===//
// VivadoQLinearKSplitPass Implementation
//===----------------------------------------------------------------------===//

namespace {

class VivadoQLinearKSplitPass 
    : public PassWrapper<VivadoQLinearKSplitPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VivadoQLinearKSplitPass)

  VivadoQLinearKSplitPass() = default;
  VivadoQLinearKSplitPass(const VivadoQLinearKSplitPass &) {}

  StringRef getArgument() const final { 
    return "vivado-qlinear-k-split"; 
  }
  
  StringRef getDescription() const final {
    return "Split vivado.qlinear operations with K dimension exceeding tile size";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vivado::VivadoDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override;

private:
  /// Collect QLinear ops that need K-dimension splitting
  void collectOpsToSplit(ModuleOp module, 
                         SmallVectorImpl<vivado_ops::QLinearOp> &opsToSplit);
  
  /// Split a single QLinear operation
  LogicalResult splitQLinearOp(vivado_ops::QLinearOp op);
  
  /// Get the K dimension size based on is_transposed attribute
  int64_t getKDimSize(vivado_ops::QLinearOp op);
  
  /// Create subview for input tensor along K dimension
  Value createInputSubview(OpBuilder &builder, Location loc, 
                           Value input, int64_t kStart, int64_t kSize,
                           bool isTransposed);
  
  /// Create subview for weight tensor along K dimension
  Value createWeightSubview(OpBuilder &builder, Location loc,
                            Value weight, int64_t kStart, int64_t kSize,
                            bool isTransposed);
  
  /// Create a partial output buffer
  Value createPartialOutputBuffer(OpBuilder &builder, Location loc,
                                  ShapedType outputType);
  
  /// Get the tile K size from configuration
  int32_t getTileKSize() const {
    return TileConfig::kDefaultTileK;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Entry Point
//===----------------------------------------------------------------------===//

void VivadoQLinearKSplitPass::runOnOperation() {
  ModuleOp module = getOperation();
  
  // Step 1: Collect all QLinear ops that need K-dimension splitting
  SmallVector<vivado_ops::QLinearOp, 8> opsToSplit;
  collectOpsToSplit(module, opsToSplit);
  
  if (opsToSplit.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No QLinear ops need K-dimension splitting\n");
    return;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << opsToSplit.size() 
                          << " QLinear ops to split\n");
  
  // Step 2: Split each collected operation
  for (vivado_ops::QLinearOp op : opsToSplit) {
    if (failed(splitQLinearOp(op))) {
      signalPassFailure();
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// Operation Collection
//===----------------------------------------------------------------------===//

void VivadoQLinearKSplitPass::collectOpsToSplit(
    ModuleOp module, SmallVectorImpl<vivado_ops::QLinearOp> &opsToSplit) {
  
  int32_t tileK = getTileKSize();
  
  module.walk([&](vivado_ops::QLinearOp op) {
    int64_t kDim = getKDimSize(op);
    
    if (kDim > tileK) {
      LLVM_DEBUG(llvm::dbgs() << "QLinear at " << op.getLoc() 
                              << " has K=" << kDim 
                              << " > TileK=" << tileK << ", will split\n");
      opsToSplit.push_back(op);
    }
  });
}

//===----------------------------------------------------------------------===//
// K Dimension Analysis
//===----------------------------------------------------------------------===//

int64_t VivadoQLinearKSplitPass::getKDimSize(vivado_ops::QLinearOp op) {
  // Get input and weight shapes
  auto inputType = op.getInput().getType().cast<ShapedType>();
  auto weightType = op.getWeight().getType().cast<ShapedType>();
  
  if (!inputType.hasStaticShape() || !weightType.hasStaticShape()) {
    // Dynamic shapes - conservatively assume needs splitting
    // In practice, this should not happen for our use case
    return std::numeric_limits<int64_t>::max();
  }
  
  // NOTE: K dimension selection depends on whether transpose-mode semantics
  // are enabled, not on whether the output memref is marked transposed.
  bool transposeMode = op.getTransposeMode();
  
  // Determine K dimension based on layout
  // is_transposed=false: input (B,L,D) or (L,D), weight (OC,IC), K=IC (last dim of input)
  // is_transposed=true:  input (B,D,L) or (D,L), weight (IC,OC), K=IC (first dim of weight)
  
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> weightShape = weightType.getShape();
  
  if (!transposeMode) {
    // Standard layout: K is the last dimension of input
    // and the second dimension of weight (IC in OC x IC)
    return inputShape.back();
  } else {
    // Transposed layout: K is the second-to-last dim of input
    // and the first dimension of weight (IC in IC x OC)
    return weightShape.front();
  }
}

//===----------------------------------------------------------------------===//
// Subview Creation Helpers
//===----------------------------------------------------------------------===//

Value VivadoQLinearKSplitPass::createInputSubview(
    OpBuilder &builder, Location loc, 
    Value input, int64_t kStart, int64_t kSize,
  bool transposeMode) {
  
  auto inputType = input.getType().cast<MemRefType>();
  ArrayRef<int64_t> shape = inputType.getShape();
  int64_t rank = shape.size();
  
  // Determine which dimension is K
  int64_t kDimIdx = transposeMode ? (rank - 2) : (rank - 1);
  
  // Build offsets, sizes, strides
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  
  for (int64_t i = 0; i < rank; ++i) {
    if (i == kDimIdx) {
      offsets[i] = builder.getIndexAttr(kStart);
      sizes.push_back(builder.getIndexAttr(kSize));
    } else {
      sizes.push_back(builder.getIndexAttr(shape[i]));
    }
  }
  
  // Infer correct result type (including strided layout) from the source.
  auto resultType = llvm::cast<MemRefType>(
      memref::SubViewOp::inferResultType(inputType, offsets, sizes, strides));

  return builder.create<memref::SubViewOp>(loc, resultType, input,
                                          offsets, sizes, strides);
}

Value VivadoQLinearKSplitPass::createWeightSubview(
    OpBuilder &builder, Location loc,
    Value weight, int64_t kStart, int64_t kSize,
  bool transposeMode) {
  
  auto weightType = weight.getType().cast<MemRefType>();
  ArrayRef<int64_t> shape = weightType.getShape();
  int64_t rank = shape.size();
  
  // Determine which dimension is K (IC dimension)
  // is_transposed=false: weight (OC, IC), K is dim 1
  // is_transposed=true:  weight (IC, OC), K is dim 0
  int64_t kDimIdx = transposeMode ? 0 : (rank - 1);
  
  // Build offsets, sizes, strides
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  
  for (int64_t i = 0; i < rank; ++i) {
    if (i == kDimIdx) {
      offsets[i] = builder.getIndexAttr(kStart);
      sizes.push_back(builder.getIndexAttr(kSize));
    } else {
      sizes.push_back(builder.getIndexAttr(shape[i]));
    }
  }
  
  // Infer correct result type (including strided layout) from the source.
  auto resultType = llvm::cast<MemRefType>(
      memref::SubViewOp::inferResultType(weightType, offsets, sizes, strides));

  return builder.create<memref::SubViewOp>(loc, resultType, weight,
                                          offsets, sizes, strides);
}

Value VivadoQLinearKSplitPass::createPartialOutputBuffer(
    OpBuilder &builder, Location loc, ShapedType outputType) {
  
  auto memrefType = outputType.cast<MemRefType>();
  return builder.create<memref::AllocOp>(loc, memrefType);
}

//===----------------------------------------------------------------------===//
// QLinear Splitting Logic
//===----------------------------------------------------------------------===//

LogicalResult VivadoQLinearKSplitPass::splitQLinearOp(vivado_ops::QLinearOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  
  // Get original operands and attributes
  Value output = op.getOutput();
  Value input = op.getInput();
  Value weight = op.getWeight();
  Value fscl = op.getFscl();
  Value iscl = op.getIscl();
  Value oscl = op.getOscl();
  Value osclInv = op.getOsclInv();
  Value wscl = op.getWscl();
  Value bscl = op.getBscl();
  Value inputZero = op.getInputZero();
  Value outputZero = op.getOutputZero();
  Value bias = op.getBias();
  
  bool transposeMode = op.getTransposeMode();
  
  // Get K dimension info
  int64_t kTotal = getKDimSize(op);
  int32_t tileK = getTileKSize();
  int64_t numSplits = (kTotal + tileK - 1) / tileK;
  
  LLVM_DEBUG(llvm::dbgs() << "Splitting QLinear with K=" << kTotal 
                          << " into " << numSplits << " tiles\n");
  
  auto outputType = output.getType().cast<MemRefType>();
  
  // Container for partial results and their outputs
  SmallVector<Value, 4> partialOutputs;
  
  // Create split QLinear operations
  for (int64_t i = 0; i < numSplits; ++i) {
    int64_t kStart = i * tileK;
    int64_t kEnd = std::min(kStart + tileK, kTotal);
    int64_t kSize = kEnd - kStart;
    
    // Create subviews for input and weight
    Value inputSlice = createInputSubview(builder, loc, input, 
                                          kStart, kSize, transposeMode);
    Value weightSlice = createWeightSubview(builder, loc, weight,
                                            kStart, kSize, transposeMode);
    
    // Optimization: For the first split, write directly to a buffer that can be reused
    // For single split case, write directly to output to avoid copy
    Value partialOutput;
    if (numSplits == 1) {
      // Single split - write directly to output, no accumulation needed
      partialOutput = output;
    } else {
      // Multiple splits - allocate temp buffer
      partialOutput = createPartialOutputBuffer(builder, loc, outputType);
    }
    partialOutputs.push_back(partialOutput);
    
    // Create the split QLinear operation
    // Note: Bias should be separated by VivadoSeparateBias pass before this pass
    // If bias exists here, only apply it to the first split to avoid double-adding
    Value splitBias = (i == 0) ? bias : Value();
    Value splitBscl = (i == 0) ? bscl : Value();
    
    // Build the new QLinear op
    builder.create<vivado_ops::QLinearOp>(
        loc,
        partialOutput,
        inputSlice,
        weightSlice,
        fscl, iscl, oscl, osclInv, wscl,
        splitBscl,
        inputZero,
        outputZero,
        splitBias,
        op.getTileMAttr(),
        op.getTileNAttr(),
        builder.getI32IntegerAttr(static_cast<int32_t>(kSize)), // tile_k for this split
        op.getBufferStrategyAttr(),
        op.getBufferIdsAttr(),
        op.getBiasBufferIdAttr(),
        op.getFuseBiasAttr(),
        op.getTransposeWeightAttr(),
        op.getHlsPragmasAttr(),
        op.getAccumulatorTypeAttr(),
        op.getRequantModeAttr(),
        op.getScalePackingAttr(),
        op.getScaleCoeModeAttr(),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr(),
        op.getLayerTypeAttr()  // Preserve layer_type
    );
  }
  
  // Now create QAdd chain to accumulate partial results
  // Skip accumulation if only one split (already written to output)
  if (numSplits == 1) {
    // Single split case - already written to output, nothing to accumulate
    LLVM_DEBUG(llvm::dbgs() << "Single split, no accumulation needed\n");
  } else {
    // Multiple splits - need to accumulate partial results
    // The first partial result is our starting accumulator
    Value accumulator = partialOutputs[0];
    
    for (size_t i = 1; i < partialOutputs.size(); ++i) {
      Value partialResult = partialOutputs[i];
      
      // For intermediate results, allocate a new buffer
      // For the last addition, we can use the original output buffer
      Value addOutput;
      if (i == partialOutputs.size() - 1) {
        // Last addition - write directly to original output
        addOutput = output;
      } else {
        // Intermediate - need temp buffer
        addOutput = createPartialOutputBuffer(builder, loc, outputType);
      }
      
      // Create QAdd operation
      // Use o_scale for both inputs since they share the same quantization scale
      // qadd(output, lhs, rhs, x_scale, y_scale, o_scale, o_scale_inv, ...)
      builder.create<vivado_ops::QAddOp>(
          loc,
          addOutput,           // output
          accumulator,         // lhs (previous accumulator)
          partialResult,       // rhs (current partial result)
          oscl,                // x_scale (same as output scale)
          oscl,                // y_scale (same as output scale)
          oscl,                // o_scale
          osclInv,             // o_scale_inv
          /*x_zero=*/outputZero,  // x_zero (same as output zero)
          /*y_zero=*/outputZero,  // y_zero (same as output zero)
          outputZero,          // o_zero
          /*fuse_into_producer=*/builder.getBoolAttr(false),
          /*vectorize=*/builder.getBoolAttr(true),
          op.getScalePackingAttr(),
          op.getScaleCoeModeAttr(),
            op.getTransposeModeAttr(),
          op.getIsTransposedAttr()
      );
      
      // Update accumulator for next iteration
      accumulator = addOutput;
    }
  }
  
  // Erase the original operation
  op.erase();
  
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createVivadoQLinearKSplitPass() {
  return std::make_unique<VivadoQLinearKSplitPass>();
}

} // namespace allo
} // namespace mlir
