/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// LowerVivadoToPYNQ Pass
// This file implements the lowering of Vivado backend quantized operations
// to low-level PYNQ hardware instruction operations.
//
// Transformations:
// - vivado.qmatmul -> pynq.buffer_alloc + pynq.data_transfer + pynq.matmul_instr
// - vivado.qlinear -> pynq.buffer_alloc + pynq.data_transfer + pynq.matmul_instr
// - vivado.qadd -> pynq.qadd
// - vivado.int_gelu -> pynq.gelu
// - vivado.int_softmax -> pynq.softmax
// - vivado.int_layernorm -> pynq.layernorm
// - vivado.qconv2d -> lowered to tiled loops + pynq ops
// - vivado.qmatmul_isqrtd -> pynq.matmul_instr with scaling
//===----------------------------------------------------------------------===//

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/VivadoOps.h"
#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQConfig.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace allo;

// Explicitly use vivado namespace for source ops
namespace vivado_ops = mlir::allo::vivado;
// Explicitly use pynq namespace for target ops
namespace pynq_ops = mlir::allo::pynq;

namespace mlir {
namespace allo {

//===----------------------------------------------------------------------===//
// Pattern: vivado.qmatmul -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQMatMulToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QMatMulOp> {
  using OpRewritePattern<vivado_ops::QMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QMatMulOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.qmatmul -> PYNQ lowering
    // This should generate:
    // 1. pynq.buffer_alloc for input, weight, output buffers
    // 2. Tiling loops for large matrices
    // 3. pynq.data_transfer (host -> device) for input and weight tiles
    // 4. pynq.matmul_instr for hardware computation
    // 5. pynq.sync for synchronization
    // 6. pynq.data_transfer (device -> host) for output tiles
    // 7. Requantization loops
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qlinear -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQLinearToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QLinearOp> {
  using OpRewritePattern<vivado_ops::QLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QLinearOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // Extract operands
    Value output = op.getOutput();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    
    // Get tile configuration
    int32_t tileM = op.getTileM();
    int32_t tileN = op.getTileN();
    int32_t tileK = op.getTileK();
    
    //===------------------------------------------------------------------===//
    // Step 1: Verify shapes can fit in buffers
    //===------------------------------------------------------------------===//
    
    auto inputType = input.getType().dyn_cast<ShapedType>();
    auto weightType = weight.getType().dyn_cast<ShapedType>();
    auto outputType = output.getType().dyn_cast<ShapedType>();
    
    if (!inputType || !weightType || !outputType) {
      return rewriter.notifyMatchFailure(op, "operands must have shaped types");
    }
    
    if (!inputType.hasStaticShape() || !weightType.hasStaticShape() || 
        !outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "dynamic shapes not yet supported");
    }
    
    // Linear: output[M, N] = input[M, K] @ weight[N, K].T + bias[N]
    // In PYNQ: output[M, N] = input[M, K] @ weight_transposed[K, N] + bias[N]
    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();
    auto outputShape = outputType.getShape();
    
    if (inputShape.size() != 2 || weightShape.size() != 2 || outputShape.size() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matrices supported");
    }
    
    int64_t M = inputShape[0];
    int64_t K = inputShape[1];
    int64_t N = weightShape[0];  // weight is [N, K]
    int64_t K_weight = weightShape[1];
    
    if (K != K_weight) {
      return rewriter.notifyMatchFailure(op, "input/weight dimension mismatch");
    }
    
    if (outputShape[0] != M || outputShape[1] != N) {
      return rewriter.notifyMatchFailure(op, "output shape mismatch");
    }
    
    // Verify tiling correctness (for now, just check divisibility)
    if (M % tileM != 0 || N % tileN != 0 || K % tileK != 0) {
      return rewriter.notifyMatchFailure(op, 
        "matrix dimensions must be divisible by tile sizes (tiling pass not implemented)");
    }
    
    // Calculate buffer capacity needed (in bytes)
    // Each tile is tileSize x tileSize x sizeof(i8)
    auto elementType = inputType.getElementType();
    if (!elementType.isInteger(8)) {
      return rewriter.notifyMatchFailure(op, "only i8 element type supported");
    }
    
    int64_t inputTileBytes = tileM * tileK * 1;  // i8 = 1 byte
    int64_t weightTileBytes = tileN * tileK * 1;
    int64_t outputTileBytes = tileM * tileN * 1;
    
    // Validate tile sizes fit in buffer capacity
    if (inputTileBytes > pynq::BufferConfig::kDefaultCapacityBytes || 
        weightTileBytes > pynq::BufferConfig::kDefaultCapacityBytes ||
        outputTileBytes > pynq::BufferConfig::kDefaultCapacityBytes) {
      return rewriter.notifyMatchFailure(op, 
        "tile size exceeds buffer capacity (adjust tiling)");
    }
    
    //===------------------------------------------------------------------===//
    // Step 2: Allocate virtual buffers for input, weight, output
    //===------------------------------------------------------------------===//
    
    // Create virtual buffers (IDs will be assigned by allocation pass later)
    auto i8Type = rewriter.getIntegerType(8);
    
    auto inputBufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, inputTileBytes);
    auto weightBufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, weightTileBytes);
    auto outputBufferType = pynq_ops::BufferType::getVirtual(
        rewriter.getContext(), i8Type, outputTileBytes);
    
    Value inputBuf = rewriter.create<pynq_ops::BufferAllocOp>(
        loc, inputBufferType, rewriter.getStringAttr("input"));
    Value weightBuf = rewriter.create<pynq_ops::BufferAllocOp>(
        loc, weightBufferType, rewriter.getStringAttr("weight"));
    Value outputBuf = rewriter.create<pynq_ops::BufferAllocOp>(
        loc, outputBufferType, rewriter.getStringAttr("output"));
    
    //===------------------------------------------------------------------===//
    // Step 3: Copy weight/quantization data from memref.global to memref.alloc
    //===------------------------------------------------------------------===//
    
    // TODO: Handle memref.global -> memref.alloc copying
    // For now, we assume weight/bias are already in allocated memrefs
    // Later passes will expand this into explicit copy operations
    
    // NOTE: Weight data movement:
    // If weight comes from memref.global (compile-time constant), we need to:
    // 1. Create a memref.alloc with same shape
    // 2. Generate copy/memcpy operation to transfer data
    // 3. Use the allocated memref for data transfer
    // 
    // For now, we assume the weight operand is already a memref.alloc or
    // function argument. The copy expansion will be handled in a separate pass.
    
    Value weightMemref = weight;  // Will be expanded to alloc + copy later
    
    //===------------------------------------------------------------------===//
    // Step 4: Generate tiling loops and data transfers
    //===------------------------------------------------------------------===//
    
    // Calculate tile counts
    int64_t numTilesM = M / tileM;
    int64_t numTilesN = N / tileN;
    int64_t numTilesK = K / tileK;
    
    // Create constant values for loop bounds and parameters
    auto createI32Const = [&](int32_t value) {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
    };
    
    auto createIndexConst = [&](int64_t value) {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(value));
    };
    
    Value c0_idx = createIndexConst(0);
    Value c1_idx = createIndexConst(1);
    Value tileM_idx = createIndexConst(tileM);
    Value tileN_idx = createIndexConst(tileN);
    Value tileK_idx = createIndexConst(tileK);
    Value M_idx = createIndexConst(M);
    Value N_idx = createIndexConst(N);
    Value K_idx = createIndexConst(K);
    
    // Create tiling loops: for m in range(0, M, tileM)
    //                       for n in range(0, N, tileN)
    //                         for k in range(0, K, tileK)
    auto forM = rewriter.create<scf::ForOp>(loc, c0_idx, M_idx, tileM_idx);
    rewriter.setInsertionPointToStart(forM.getBody());
    
    auto forN = rewriter.create<scf::ForOp>(loc, c0_idx, N_idx, tileN_idx);
    rewriter.setInsertionPointToStart(forN.getBody());
    
    auto forK = rewriter.create<scf::ForOp>(loc, c0_idx, K_idx, tileK_idx);
    rewriter.setInsertionPointToStart(forK.getBody());
    
    // Get loop induction variables
    Value m_idx = forM.getInductionVar();
    Value n_idx = forN.getInductionVar();
    Value k_idx = forK.getInductionVar();
    
    //===------------------------------------------------------------------===//
    // Step 5: Generate data transfers (PS -> PL)
    //===------------------------------------------------------------------===//
    
    // TODO: Generate subview/slice operations to extract tiles from input/weight
    // For now, we'll generate placeholder data transfer operations
    // The actual slicing will be added in next iteration
    
    // Calculate transfer parameters
    Value inputTileCount = createI32Const(1);  // Number of tiles in this transfer
    Value weightTileCount = createI32Const(1);
    Value outputTileCount = createI32Const(1);
    
    // Calculate total package numbers (for DMA)
    // 1 package = 32 bytes = 32 x i8
    int64_t inputPkgNum = pynq::calculatePackageCount(inputTileBytes);
    int64_t weightPkgNum = pynq::calculatePackageCount(weightTileBytes);
    int64_t outputPkgNum = pynq::calculatePackageCount(outputTileBytes);
    
    Value inputPkgNumVal = createI32Const(inputPkgNum);
    Value weightPkgNumVal = createI32Const(weightPkgNum);
    Value outputPkgNumVal = createI32Const(outputPkgNum);
    
    Value direction_load = createI32Const(pynq::DMAConfig::kDirectionLoad);   // host -> device
    Value direction_store = createI32Const(pynq::DMAConfig::kDirectionStore);  // device -> host
    
    // Note: We need buffer IDs as i32, but virtual buffers don't have IDs yet
    // We'll use placeholder values that will be replaced after buffer allocation
    Value inputBufId = createI32Const(0);   // Placeholder, will be updated
    Value weightBufId = createI32Const(1);  // Placeholder
    Value outputBufId = createI32Const(2);  // Placeholder
    
    // TODO: Generate subview to extract input[m:m+tileM, k:k+tileK]
    // For now, use the full input memref as placeholder
    // rewriter.create<pynq_ops::DataTransferOp>(
    //     loc, input_tile_subview, inputBufId, inputTileCount, 
    //     inputPkgNumVal, direction_load);
    
    // TODO: Generate subview to extract weight[n:n+tileN, k:k+tileK]
    // rewriter.create<pynq_ops::DataTransferOp>(
    //     loc, weight_tile_subview, weightBufId, weightTileCount,
    //     weightPkgNumVal, direction_load);
    
    //===------------------------------------------------------------------===//
    // Step 6: Generate matrix multiplication instruction
    //===------------------------------------------------------------------===//
    
    Value reduceK = createI32Const(tileK);
    Value headCount = createI32Const(0);      // No multi-head attention
    Value headTileAxis = createI32Const(0);
    Value enableBias = createI32Const(op.getFuseBias() ? 1 : 0);
    Value enableTranspose = createI32Const(op.getTransposeWeight() ? 1 : 0);
    
    rewriter.create<pynq_ops::MatMulInstrOp>(
        loc,
        inputBufId,      // input_buffer_id (RS1)
        weightBufId,     // weight_buffer_id (RS2) 
        outputBufId,     // output_buffer_id (RD)
        inputTileCount,  // input_tile_count
        weightTileCount, // weight_tile_count
        reduceK,         // reduce_k
        headCount,       // head_count
        headTileAxis,    // head_tile_axis
        enableBias,      // enable_bias
        enableTranspose  // enable_transpose
    );
    
    //===------------------------------------------------------------------===//
    // Step 7: Generate synchronization
    //===------------------------------------------------------------------===//
    
    rewriter.create<pynq_ops::SyncOp>(loc);
    
    //===------------------------------------------------------------------===//
    // Step 8: Generate data transfer back (PL -> PS)
    //===------------------------------------------------------------------===//
    
    // TODO: Generate subview to extract output[m:m+tileM, n:n+tileN]
    // rewriter.create<pynq_ops::DataTransferOp>(
    //     loc, output_tile_subview, outputBufId, outputTileCount,
    //     outputPkgNumVal, direction_store);
    
    //===------------------------------------------------------------------===//
    // Step 9: Quantization parameter handling (TODO)
    //===------------------------------------------------------------------===//
    
    // TODO: Handle quantization parameters (fscl, iscl, oscl, wscl, bscl)
    // These need to be:
    // 1. Extracted from the vivado.qlinear operands
    // 2. Transferred to device if needed
    // 3. Applied in requantization loops after matmul
    //
    // For now, we add comments as placeholders for visibility:
    
    rewriter.setInsertionPointAfter(forM);
    
    // Add comment marker for quantization handling
    // (In real implementation, this would be requantization loops)
    rewriter.create<scf::YieldOp>(loc);
    
    //===------------------------------------------------------------------===//
    // Step 10: Bias handling (TODO)
    //===------------------------------------------------------------------===//
    
    // TODO: Handle bias if present
    // Options:
    // 1. Fuse into matmul instruction (if enable_bias=1 and hardware supports)
    // 2. Generate separate pynq.qadd operation after matmul
    // 3. Generate explicit addition loops
    //
    // For now, we note this with a comment:
    if (bias) {
      // Bias handling deferred:
      // - If fuse_bias=true: bias should be loaded to buffer (default: pynq::BiasConfig::kDefaultBiasBufferId) and 
      //   enable_bias flag set in matmul_instr
      // - If fuse_bias=false: generate pynq.qadd or explicit loops
      // - Bias quantization parameters (bscl_sign, bscl_coe, bscl_rshift)
      //   need to be handled
    }
    
    //===------------------------------------------------------------------===//
    // Step 11: Remove original vivado.qlinear op
    //===------------------------------------------------------------------===//
    
    rewriter.eraseOp(op);
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qadd -> pynq.qadd
//===----------------------------------------------------------------------===//

struct VivadoQAddToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QAddOp> {
  using OpRewritePattern<vivado_ops::QAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QAddOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.qadd -> pynq.qadd lowering
    // Need to:
    // 1. Generate buffer allocation if needed
    // 2. Generate pynq.qadd instruction
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.int_gelu -> pynq.gelu
//===----------------------------------------------------------------------===//

struct VivadoIntGELUToPYNQPattern 
    : public OpRewritePattern<vivado_ops::IntGELUOp> {
  using OpRewritePattern<vivado_ops::IntGELUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::IntGELUOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.int_gelu -> pynq.gelu lowering
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.int_softmax -> pynq.softmax
//===----------------------------------------------------------------------===//

struct VivadoIntSoftmaxToPYNQPattern 
    : public OpRewritePattern<vivado_ops::IntSoftmaxOp> {
  using OpRewritePattern<vivado_ops::IntSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::IntSoftmaxOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.int_softmax -> pynq.softmax lowering
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.int_layernorm -> pynq.layernorm
//===----------------------------------------------------------------------===//

struct VivadoIntLayerNormToPYNQPattern 
    : public OpRewritePattern<vivado_ops::IntLayerNormOp> {
  using OpRewritePattern<vivado_ops::IntLayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::IntLayerNormOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.int_layernorm -> pynq.layernorm lowering
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qconv2d -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQConv2dToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QConv2dOp> {
  using OpRewritePattern<vivado_ops::QConv2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QConv2dOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.qconv2d -> PYNQ lowering
    // Conv2d needs to be transformed to im2col + matmul pattern
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pattern: vivado.qmatmul_isqrtd -> PYNQ instructions
//===----------------------------------------------------------------------===//

struct VivadoQMatMulIsqrtDToPYNQPattern 
    : public OpRewritePattern<vivado_ops::QMatMulIsqrtDOp> {
  using OpRewritePattern<vivado_ops::QMatMulIsqrtDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vivado_ops::QMatMulIsqrtDOp op,
                                 PatternRewriter &rewriter) const override {
    // TODO: Implement vivado.qmatmul_isqrtd -> PYNQ lowering
    // Similar to qmatmul but includes 1/sqrt(d) scaling for attention
    
    return failure(); // Not implemented yet
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

bool applyLowerVivadoToPYNQ(ModuleOp &module, MLIRContext *context) {
  // Setup rewrite patterns
  RewritePatternSet patterns(context);
  
  // Add all lowering patterns
  patterns.add<VivadoQMatMulToPYNQPattern>(context);
  patterns.add<VivadoQLinearToPYNQPattern>(context);
  patterns.add<VivadoQAddToPYNQPattern>(context);
  patterns.add<VivadoIntGELUToPYNQPattern>(context);
  patterns.add<VivadoIntSoftmaxToPYNQPattern>(context);
  patterns.add<VivadoIntLayerNormToPYNQPattern>(context);
  patterns.add<VivadoQConv2dToPYNQPattern>(context);
  patterns.add<VivadoQMatMulIsqrtDToPYNQPattern>(context);

  return !failed(applyPatternsAndFoldGreedily(module, std::move(patterns)));
}

struct LowerVivadoToPYNQPass
    : public PassWrapper<LowerVivadoToPYNQPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVivadoToPYNQPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Register PYNQ dialect and other dependencies
    registry.insert<pynq_ops::PYNQDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const final { return "lower-vivado-to-pynq"; }
  
  StringRef getDescription() const final {
    return "Lower Vivado backend ops to PYNQ hardware instructions";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Apply patterns using greedy rewrite
    if (!applyLowerVivadoToPYNQ(module, context)) {
      signalPassFailure();
      return;
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLowerVivadoToPYNQPass() {
  return std::make_unique<LowerVivadoToPYNQPass>();
}

} // namespace allo
} // namespace mlir
