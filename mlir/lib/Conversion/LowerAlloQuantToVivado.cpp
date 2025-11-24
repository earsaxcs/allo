/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// LowerAlloQuantToVivado Pass
// This file implements the lowering of platform-agnostic Allo quantization
// operations to Vivado/PYNQ backend-specific operations.
//
// Transformations:
// - allo.qmatmul -> vivado.qmatmul (1:1 operand mapping + default attributes)
// - allo.qlinear -> vivado.qlinear (with bias fusion)
// - allo.qadd -> vivado.qadd (with fusion hints)
// - allo.int_gelu -> vivado.int_gelu (LUT-based)
// - allo.int_softmax -> vivado.int_softmax (LUT-based)
// - allo.int_layernorm -> vivado.int_layernorm (approx rsqrt)
//===----------------------------------------------------------------------===//

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/VivadoOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace allo;

// Explicitly use allo namespace for source ops
namespace allo_ops = mlir::allo;
// Explicitly use vivado namespace for target ops
namespace vivado_ops = mlir::allo::vivado;

namespace mlir {
namespace allo {

//===----------------------------------------------------------------------===//
// Pattern: allo.qmatmul -> vivado.qmatmul
//===----------------------------------------------------------------------===//

struct QMatMulLoweringPattern : public OpRewritePattern<allo_ops::QMatMulOp> {
  using OpRewritePattern<allo_ops::QMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::QMatMulOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Extract all operands (preserve exact semantics)
    Value output = op.getOutput();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // Input scale parameters
    Value xScaleSign = op.getXScaleSign();
    Value xScaleCoe = op.getXScaleCoe();
    Value xScaleRshift = op.getXScaleRshift();
    
    // Weight scale parameters
    Value yScaleSign = op.getYScaleSign();
    Value yScaleCoe = op.getYScaleCoe();
    Value yScaleRshift = op.getYScaleRshift();
    
    // Output scale parameters
    Value oScaleSign = op.getOScaleSign();
    Value oScaleCoe = op.getOScaleCoe();
    Value oScaleRshift = op.getOScaleRshift();
    
    // Optional zero points
    Value xZero = op.getXZero();
    Value yZero = op.getYZero();
    Value oZero = op.getOZero();

    // Create vivado.qmatmul op with default backend attributes
    rewriter.replaceOpWithNewOp<vivado_ops::QMatMulOp>(
        op, output, lhs, rhs,
        xScaleSign, xScaleCoe, xScaleRshift,
        yScaleSign, yScaleCoe, yScaleRshift,
        oScaleSign, oScaleCoe, oScaleRshift,
        xZero, yZero, oZero,
        // Backend-specific attributes with defaults
        rewriter.getI32IntegerAttr(32),  // tile_m
        rewriter.getI32IntegerAttr(32),  // tile_n
        rewriter.getI32IntegerAttr(32),  // tile_k
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids (optional)
        rewriter.getBoolAttr(false),     // enable_bias
        rewriter.getBoolAttr(false),     // transpose_weight
        nullptr,                         // hls_pragmas (optional)
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline") // requant_mode
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: allo.qlinear -> vivado.qlinear
//===----------------------------------------------------------------------===//

struct QLinearLoweringPattern : public OpRewritePattern<allo_ops::QLinearOp> {
  using OpRewritePattern<allo_ops::QLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::QLinearOp op,
                                 PatternRewriter &rewriter) const override {
    // Extract all operands
    Value output = op.getOutput();
    Value input = op.getInput();
    Value weight = op.getWeight();
    
    // Scale parameters (5 sets for fscl, iscl, oscl, wscl, bscl)
    Value fscaleSign = op.getFsclSign();
    Value fscaleCoe = op.getFsclCoe();
    Value fscaleRshift = op.getFsclRshift();
    
    Value iscaleSign = op.getIsclSign();
    Value iscaleCoe = op.getIsclCoe();
    Value iscaleRshift = op.getIsclRshift();
    
    Value oscaleSign = op.getOsclSign();
    Value oscaleCoe = op.getOsclCoe();
    Value oscaleRshift = op.getOsclRshift();
    
    Value wscaleSign = op.getWsclSign();
    Value wscaleCoe = op.getWsclCoe();
    Value wscaleRshift = op.getWsclRshift();
    
    Value bscaleSign = op.getBsclSign();
    Value bscaleCoe = op.getBsclCoe();
    Value bscaleRshift = op.getBsclRshift();
    
    // Optional parameters
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    Value bias = op.getBias();

    // Create vivado.qlinear op with default backend attributes
    rewriter.replaceOpWithNewOp<vivado_ops::QLinearOp>(
        op, output, input, weight,
        fscaleSign, fscaleCoe, fscaleRshift,
        iscaleSign, iscaleCoe, iscaleRshift,
        oscaleSign, oscaleCoe, oscaleRshift,
        wscaleSign, wscaleCoe, wscaleRshift,
        bscaleSign, bscaleCoe, bscaleRshift,
        inputZero, outputZero, bias,
        // Backend-specific attributes
        rewriter.getI32IntegerAttr(32),  // tile_m
        rewriter.getI32IntegerAttr(32),  // tile_n
        rewriter.getI32IntegerAttr(32),  // tile_k
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids (optional)
        rewriter.getI32IntegerAttr(7),   // bias_buffer_id
        rewriter.getBoolAttr(true),      // fuse_bias
        rewriter.getBoolAttr(false),     // transpose_weight
        nullptr,                         // hls_pragmas (optional)
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline") // requant_mode
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: allo.qadd -> vivado.qadd
//===----------------------------------------------------------------------===//

struct QAddLoweringPattern : public OpRewritePattern<allo_ops::QAddOp> {
  using OpRewritePattern<allo_ops::QAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::QAddOp op,
                                 PatternRewriter &rewriter) const override {
    Value output = op.getOutput();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    Value xScaleSign = op.getXScaleSign();
    Value xScaleCoe = op.getXScaleCoe();
    Value xScaleRshift = op.getXScaleRshift();
    
    Value yScaleSign = op.getYScaleSign();
    Value yScaleCoe = op.getYScaleCoe();
    Value yScaleRshift = op.getYScaleRshift();
    
    Value oScaleSign = op.getOScaleSign();
    Value oScaleCoe = op.getOScaleCoe();
    Value oScaleRshift = op.getOScaleRshift();
    
    Value xZero = op.getXZero();
    Value yZero = op.getYZero();
    Value oZero = op.getOZero();

    // Create vivado.qadd op with vectorization hints
    rewriter.replaceOpWithNewOp<vivado_ops::QAddOp>(
        op, output, lhs, rhs,
        xScaleSign, xScaleCoe, xScaleRshift,
        yScaleSign, yScaleCoe, yScaleRshift,
        oScaleSign, oScaleCoe, oScaleRshift,
        xZero, yZero, oZero,
        // Backend-specific attributes
        rewriter.getBoolAttr(false),  // fuse_into_producer
        rewriter.getBoolAttr(true)    // vectorize
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: allo.int_gelu -> vivado.int_gelu
//===----------------------------------------------------------------------===//

struct IntGELULoweringPattern : public OpRewritePattern<allo_ops::IntGELUOp> {
  using OpRewritePattern<allo_ops::IntGELUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::IntGELUOp op,
                                 PatternRewriter &rewriter) const override {
    Value output = op.getOutput();
    Value input = op.getInput();
    
    // Extract all 4 scale groups from Allo IntGELUOp
    // 1. input_scale -> Vivado iscl
    Value inputScaleSign = op.getInputScaleSign();
    Value inputScaleCoe = op.getInputScaleCoe();
    Value inputScaleRshift = op.getInputScaleRshift();
    
    // 2. gelu_scale -> Vivado gscl
    Value geluScaleSign = op.getGeluScaleSign();
    Value geluScaleCoe = op.getGeluScaleCoe();
    Value geluScaleRshift = op.getGeluScaleRshift();
    
    // 3. fused_scale -> Vivado fscl
    Value fusedScaleSign = op.getFusedScaleSign();
    Value fusedScaleCoe = op.getFusedScaleCoe();
    Value fusedScaleRshift = op.getFusedScaleRshift();
    
    // 4. output_scale -> Vivado oscl
    Value outputScaleSign = op.getOutputScaleSign();
    Value outputScaleCoe = op.getOutputScaleCoe();
    Value outputScaleRshift = op.getOutputScaleRshift();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();

    // Create vivado.int_gelu with LUT implementation and all 4 scale groups
    rewriter.replaceOpWithNewOp<vivado_ops::IntGELUOp>(
        op, output, input,
        inputScaleSign, inputScaleCoe, inputScaleRshift,
        geluScaleSign, geluScaleCoe, geluScaleRshift,
        fusedScaleSign, fusedScaleCoe, fusedScaleRshift,
        outputScaleSign, outputScaleCoe, outputScaleRshift,
        inputZero, outputZero,
        // Backend-specific attributes
        rewriter.getStringAttr("lut")  // implementation: "lut" or "polynomial"
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: allo.int_softmax -> vivado.int_softmax
//===----------------------------------------------------------------------===//

struct IntSoftmaxLoweringPattern : public OpRewritePattern<allo_ops::IntSoftmaxOp> {
  using OpRewritePattern<allo_ops::IntSoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::IntSoftmaxOp op,
                                 PatternRewriter &rewriter) const override {
    Value output = op.getOutput();
    Value input = op.getInput();
    
    Value inputScaleSign = op.getInputScaleSign();
    Value inputScaleCoe = op.getInputScaleCoe();
    Value inputScaleRshift = op.getInputScaleRshift();
    
    Value softmaxScaleSign = op.getSoftmaxScaleSign();
    Value softmaxScaleCoe = op.getSoftmaxScaleCoe();
    Value softmaxScaleRshift = op.getSoftmaxScaleRshift();

    Value fusedScaleSign = op.getFusedScaleSign();
    Value fusedScaleCoe = op.getFusedScaleCoe();
    Value fusedScaleRshift = op.getFusedScaleRshift();

    Value outputScaleSign = op.getOutputScaleSign();
    Value outputScaleCoe = op.getOutputScaleCoe();
    Value outputScaleRshift = op.getOutputScaleRshift();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    
    // TODO：add axis parameter
    int64_t axis = 1; // op.getAxis();

    // Create vivado.int_softmax with LUT implementation
    rewriter.replaceOpWithNewOp<vivado_ops::IntSoftmaxOp>(
        op, output, input,
        inputScaleSign, inputScaleCoe, inputScaleRshift,
        softmaxScaleSign, softmaxScaleCoe, softmaxScaleRshift,
        outputScaleSign, outputScaleCoe, outputScaleRshift,
        fusedScaleSign, fusedScaleCoe, fusedScaleRshift,
        inputZero, outputZero,
        rewriter.getI64IntegerAttr(axis),
        // Backend-specific attributes
        rewriter.getStringAttr("lut")  // implementation
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: allo.int_layernorm -> vivado.int_layernorm
//===----------------------------------------------------------------------===//

struct IntLayerNormLoweringPattern : public OpRewritePattern<allo_ops::IntLayerNormOp> {
  using OpRewritePattern<allo_ops::IntLayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::IntLayerNormOp op,
                                 PatternRewriter &rewriter) const override {
    Value output = op.getOutput();
    Value input = op.getInput();
    Value biasInt = op.getBiasInt();
    
    Value inputScaleSign = op.getInputScaleSign();
    Value inputScaleCoe = op.getInputScaleCoe();
    Value inputScaleRshift = op.getInputScaleRshift();

    Value layernormScaleSign = op.getLayernormScaleSign();
    Value layernormScaleCoe = op.getLayernormScaleCoe();
    Value layernormScaleRshift = op.getLayernormScaleRshift();
    
    Value biasScaleSign = op.getBiasScaleSign();
    Value biasScaleCoe = op.getBiasScaleCoe();
    Value biasScaleRshift = op.getBiasScaleRshift();

    Value fusedScaleSign = op.getFusedScaleSign();
    Value fusedScaleCoe = op.getFusedScaleCoe();
    Value fusedScaleRshift = op.getFusedScaleRshift();

    Value outputScaleSign = op.getOutputScaleSign();
    Value outputScaleCoe = op.getOutputScaleCoe();
    Value outputScaleRshift = op.getOutputScaleRshift();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    
    // TODO: add eps parameter
    float eps = 1e-5; // op.getEps();

    // Create vivado.int_layernorm with approx rsqrt
    rewriter.replaceOpWithNewOp<vivado_ops::IntLayerNormOp>(
        op, output, input, biasInt,
        inputScaleSign, inputScaleCoe, inputScaleRshift,
        layernormScaleSign, layernormScaleCoe, layernormScaleRshift,
        biasScaleSign, biasScaleCoe, biasScaleRshift,
        fusedScaleSign, fusedScaleCoe, fusedScaleRshift,
        outputScaleSign, outputScaleCoe, outputScaleRshift,
        inputZero, outputZero,
        rewriter.getF32FloatAttr(eps),
        // Backend-specific attributes
        rewriter.getStringAttr("approx")  // rsqrt_method
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: allo.qconv2d -> vivado.qconv2d
//===----------------------------------------------------------------------===//

struct QConv2dLoweringPattern : public OpRewritePattern<allo_ops::QConv2dOp> {
  using OpRewritePattern<allo_ops::QConv2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::QConv2dOp op,
                                 PatternRewriter &rewriter) const override {
    Value output = op.getOutput();
    Value input = op.getInput();
    Value filter = op.getFilter();
    
    Value fscaleSign = op.getFsclSign();
    Value fscaleCoe = op.getFsclCoe();
    Value fscaleRshift = op.getFsclRshift();
    
    Value iscaleSign = op.getIsclSign();
    Value iscaleCoe = op.getIsclCoe();
    Value iscaleRshift = op.getIsclRshift();
    
    Value oscaleSign = op.getOsclSign();
    Value oscaleCoe = op.getOsclCoe();
    Value oscaleRshift = op.getOsclRshift();
    
    Value wscaleSign = op.getWsclSign();
    Value wscaleCoe = op.getWsclCoe();
    Value wscaleRshift = op.getWsclRshift();
    
    Value bscaleSign = op.getBsclSign();
    Value bscaleCoe = op.getBsclCoe();
    Value bscaleRshift = op.getBsclRshift();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    Value bias = op.getBias();
    
    auto stride = op.getStride();

    rewriter.replaceOpWithNewOp<vivado_ops::QConv2dOp>(
        op, output, input, filter,
        fscaleSign, fscaleCoe, fscaleRshift,
        iscaleSign, iscaleCoe, iscaleRshift,
        oscaleSign, oscaleCoe, oscaleRshift,
        wscaleSign, wscaleCoe, wscaleRshift,
        bscaleSign, bscaleCoe, bscaleRshift,
        inputZero, outputZero, bias,
        rewriter.getDenseI64ArrayAttr(stride),
        // Backend-specific attributes
        rewriter.getI32IntegerAttr(32),  // tile_h
        rewriter.getI32IntegerAttr(32),  // tile_w
        rewriter.getI32IntegerAttr(32),  // tile_c
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids
        rewriter.getBoolAttr(true),      // fuse_bias
        nullptr,                         // hls_pragmas
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline") // requant_mode
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: allo.qmatmul_isqrtd -> vivado.qmatmul_isqrtd
//===----------------------------------------------------------------------===//

struct QMatMulIsqrtDLoweringPattern : public OpRewritePattern<allo_ops::QMatMulIsqrtDOp> {
  using OpRewritePattern<allo_ops::QMatMulIsqrtDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::QMatMulIsqrtDOp op,
                                 PatternRewriter &rewriter) const override {
    Value output = op.getOutput();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    Value xScaleSign = op.getXScaleSign();
    Value xScaleCoe = op.getXScaleCoe();
    Value xScaleRshift = op.getXScaleRshift();
    
    Value yScaleSign = op.getYScaleSign();
    Value yScaleCoe = op.getYScaleCoe();
    Value yScaleRshift = op.getYScaleRshift();
    
    Value oScaleSign = op.getOScaleSign();
    Value oScaleCoe = op.getOScaleCoe();
    Value oScaleRshift = op.getOScaleRshift();
    
    Value xZero = op.getXZero();
    Value yZero = op.getYZero();
    Value oZero = op.getOZero();

    rewriter.replaceOpWithNewOp<vivado_ops::QMatMulIsqrtDOp>(
        op, output, lhs, rhs,
        xScaleSign, xScaleCoe, xScaleRshift,
        yScaleSign, yScaleCoe, yScaleRshift,
        oScaleSign, oScaleCoe, oScaleRshift,
        xZero, yZero, oZero,
        // Backend-specific attributes
        rewriter.getI32IntegerAttr(32),  // tile_m
        rewriter.getI32IntegerAttr(32),  // tile_n
        rewriter.getI32IntegerAttr(32),  // tile_k
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids
        rewriter.getBoolAttr(false),     // enable_bias
        rewriter.getBoolAttr(false),     // transpose_weight
        nullptr,                         // hls_pragmas
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline") // requant_mode
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

bool applyLowerAlloQuantToVivado(ModuleOp &module, MLIRContext *context) {
  // Setup rewrite patterns
  RewritePatternSet patterns(context);
  patterns.add<QMatMulLoweringPattern>(context);
  patterns.add<QMatMulIsqrtDLoweringPattern>(context);
  patterns.add<QLinearLoweringPattern>(context);
  patterns.add<QConv2dLoweringPattern>(context);
  patterns.add<QAddLoweringPattern>(context);
  patterns.add<IntGELULoweringPattern>(context);
  patterns.add<IntSoftmaxLoweringPattern>(context);
  patterns.add<IntLayerNormLoweringPattern>(context);
  patterns.add<QConv2dLoweringPattern>(context);
  patterns.add<QMatMulIsqrtDLoweringPattern>(context);

  return !failed(applyPatternsAndFoldGreedily(module, std::move(patterns)));
}

struct LowerAlloQuantToVivadoPass
    : public PassWrapper<LowerAlloQuantToVivadoPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAlloQuantToVivadoPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Register Vivado dialect as dependency
    registry.insert<vivado_ops::VivadoDialect>();
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const final { return "lower-allo-quant-to-vivado"; }
  
  StringRef getDescription() const final {
    return "Lower Allo quantized ops to Vivado backend ops";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Apply patterns using greedy rewrite
    if (!applyLowerAlloQuantToVivado(module, context)) {
      signalPassFailure();
      return;
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLowerAlloQuantToVivadoPass() {
  return std::make_unique<LowerAlloQuantToVivadoPass>();
}

} // namespace allo
} // namespace mlir
