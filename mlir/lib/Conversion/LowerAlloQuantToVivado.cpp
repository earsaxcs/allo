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
#include "allo/Dialect/PYNQConfig.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace allo;

// Explicitly use allo namespace for source ops
namespace allo_ops = mlir::allo;
// Explicitly use vivado namespace for target ops
namespace vivado_ops = mlir::allo::vivado;

namespace mlir {
namespace allo {

//===----------------------------------------------------------------------===//
// Scale Packing Helper Functions
//===----------------------------------------------------------------------===//

// Default configuration for Vivado backend
static constexpr const char* kScalePackingAttr = "8,24,8,16,16,0";
static constexpr const char* kScaleCoeModeAttr = "Tail";
static constexpr int kExpectedAlloFixedBits = 17;  // Expected fixed_bits from Allo ops

// Get fixed_bits from module attribute or use default
// Returns the fixed_bits value and emits a warning if validation fails
static int getAndValidateFixedBits(ModuleOp module, bool &validationFailed) {
  validationFailed = false;
  
  // Try to read allo.quant.fixed_bits from module attribute
  if (auto fixedBitsAttr = module->getAttrOfType<IntegerAttr>("allo.quant.fixed_bits")) {
    int fixed_bits = fixedBitsAttr.getInt();
    
    // Validate: must match kExpectedAlloFixedBits
    if (fixed_bits != kExpectedAlloFixedBits) {
      module.emitWarning() 
          << "Module attribute 'allo.quant.fixed_bits' = " << fixed_bits
          << " does not match expected value " << kExpectedAlloFixedBits
          << ". This may cause incorrect scale conversion. "
          << "Please ensure SCALE_FIXED_BITS in Python matches kExpectedAlloFixedBits in C++.";
      validationFailed = true;
    }
    
    return fixed_bits;
  }
  
  // No attribute found - emit warning and use default
  module.emitWarning()
      << "Module attribute 'allo.quant.fixed_bits' not found. "
      << "Using default value " << kExpectedAlloFixedBits << ". "
      << "Make sure SCALE_FIXED_BITS is properly passed from Python frontend.";
  
  return kExpectedAlloFixedBits;
}

// Parse ScalePackingAttr string "sign_bits,sign_offset,rshift_bits,rshift_offset,coe_bits,coe_offset"
struct ScalePackingConfig {
  int sign_bits;
  int sign_offset;
  int rshift_bits;
  int rshift_offset;
  int coe_bits;
  int coe_offset;
  
  static ScalePackingConfig parse(const std::string &attr) {
    ScalePackingConfig config;
    std::vector<int> values;
    std::string str = attr;
    size_t pos = 0;
    
    // Parse comma-separated integers
    while ((pos = str.find(',')) != std::string::npos) {
      values.push_back(std::stoi(str.substr(0, pos)));
      str.erase(0, pos + 1);
    }
    values.push_back(std::stoi(str));  // Last value
    
    if (values.size() != 6) {
      // Invalid format, use defaults
      return {8, 24, 8, 16, 16, 0};
    }
    
    config.sign_bits = values[0];
    config.sign_offset = values[1];
    config.rshift_bits = values[2];
    config.rshift_offset = values[3];
    config.coe_bits = values[4];
    config.coe_offset = values[5];
    
    return config;
  }
};

// Pack sign, coe, rshift scalars into a single i32 value at compile time
// Dynamically parses ScalePackingAttr and handles Tail/Full mode
//
// Parameters:
//   - sign: sign value
//   - coe: coefficient value (assumed to be from [0.5, 1.0) * 2^fixed_bits)
//   - rshift: right shift value
//   - packingAttr: format string like "8,24,8,16,16,0"
//   - coeMode: "Tail" or "Full"
//   - alloFixedBits: number of bits used by Allo to represent [0.5, 1.0) (typically 17)
//
// Tail mode: Extract the variant part (low bits) after the leading 1
//   - Input: 17-bit coe (e.g., 0b1_0100000000000000 for 0.625)
//   - Output: Extract low 16 bits (coe_bits) -> 0b0100000000000000
//
// Full mode: Extract the full mantissa including the leading 1
//   - Input: 17-bit coe
//   - Output: Extract high 16 bits (coe_bits) -> shift right by (17 - 16) = 1
static uint32_t packScaleValues(int8_t sign, uint32_t coe, int16_t rshift,
                                  const std::string &packingAttr,
                                  const std::string &coeMode,
                                  int alloFixedBits) {
  auto config = ScalePackingConfig::parse(packingAttr);
  if (config.sign_bits == 0) {
    llvm::errs() << "Invalid ScalePackingAttr in Vivado Dialect entry: sign_bits = 0. ";
  }
  
  // TODO: Need Validation
  // Extract coe based on mode
  uint32_t coe_packed;
  uint32_t shift_offset = 0;
  if (coeMode == "Tail") {
    // Tail mode: coe_bits represents the variant part (low bits)
    // Extract low coe_bits from the fixed-point representation
    shift_offset = 1;
  } else {
    // Full mode: coe_bits represents the full mantissa including leading 1
    // Extract high coe_bits from the fixed-point representation
    shift_offset = 0;
  }
  int shift_amount = alloFixedBits - config.coe_bits - shift_offset;
  rshift -= shift_amount;
  if (shift_amount > 0) {
    coe_packed = coe >> shift_amount;
  } else if (shift_amount < 0) {
    coe_packed = coe << (-shift_amount);
  }
  else {
    coe_packed = coe;
  }
  coe_packed &= (1u << config.coe_bits) - 1;
  
  // Pack all components according to config
  uint32_t packed = 0;
  uint32_t sign_mask = (1u << config.sign_bits) - 1;
  uint32_t rshift_mask = (1u << config.rshift_bits) - 1;
  
  if (config.sign_bits == 1) {
    sign = (sign < 0) ? 1 : 0;
  }

  packed |= ((sign & sign_mask) << config.sign_offset);
  packed |= ((rshift & rshift_mask) << config.rshift_offset);
  packed |= (coe_packed << config.coe_offset);
  
  return packed;
}

// Convert a global memref<...xi8/i16> to packed memref<...xi32>
// This creates a new global constant with packed values
static Value convertGlobalScaleToPacked(
    PatternRewriter &rewriter, Location loc, ModuleOp module,
    Value signGlobal, Value coeGlobal, Value rshiftGlobal,
    const std::string &baseName) {
  
  // Check if inputs are memref.get_global operations
  auto signGetGlobal = signGlobal.getDefiningOp<memref::GetGlobalOp>();
  auto coeGetGlobal = coeGlobal.getDefiningOp<memref::GetGlobalOp>();
  auto rshiftGetGlobal = rshiftGlobal.getDefiningOp<memref::GetGlobalOp>();
  
  if (!signGetGlobal || !coeGetGlobal || !rshiftGetGlobal) {
    // Not global constants, return null - will need runtime packing
    return Value();
  }
  
  // Find the global ops
  auto signGlobalOp = module.lookupSymbol<memref::GlobalOp>(signGetGlobal.getName());
  auto coeGlobalOp = module.lookupSymbol<memref::GlobalOp>(coeGetGlobal.getName());
  auto rshiftGlobalOp = module.lookupSymbol<memref::GlobalOp>(rshiftGetGlobal.getName());
  
  if (!signGlobalOp || !coeGlobalOp || !rshiftGlobalOp) {
    return Value();
  }
  
  // Get initial values
  auto signAttr = signGlobalOp.getInitialValue();
  auto coeAttr = coeGlobalOp.getInitialValue();
  auto rshiftAttr = rshiftGlobalOp.getInitialValue();
  
  if (!signAttr || !coeAttr || !rshiftAttr) {
    return Value();
  }
  
  // Extract DenseElementsAttr
  auto signDense = signAttr->dyn_cast<DenseElementsAttr>();
  auto coeDense = coeAttr->dyn_cast<DenseElementsAttr>();
  auto rshiftDense = rshiftAttr->dyn_cast<DenseElementsAttr>();
  
  if (!signDense || !coeDense || !rshiftDense) {
    return Value();
  }
  
  // Get shapes and verify they match
  auto signType = signGlobalOp.getType().cast<MemRefType>();
  auto coeType = coeGlobalOp.getType().cast<MemRefType>();
  auto rshiftType = rshiftGlobalOp.getType().cast<MemRefType>();
  
  if (signType.getShape() != coeType.getShape() ||
      signType.getShape() != rshiftType.getShape()) {
    return Value();
  }
  
  // Create packed values
  SmallVector<uint32_t> packedValues;
  auto signValues = signDense.getValues<int8_t>();
  auto coeValues = coeDense.getValues<uint16_t>();
  auto rshiftValues = rshiftDense.getValues<int16_t>();
  
  auto signIt = signValues.begin();
  auto coeIt = coeValues.begin();
  auto rshiftIt = rshiftValues.begin();
  
  while (signIt != signValues.end() && coeIt != coeValues.end() && rshiftIt != rshiftValues.end()) {
    uint32_t packed = packScaleValues(*signIt, *coeIt, *rshiftIt,
                                      kScalePackingAttr, kScaleCoeModeAttr, kExpectedAlloFixedBits);
    packedValues.push_back(packed);
    ++signIt;
    ++coeIt;
    ++rshiftIt;
  }
  
  // Create new global with packed i32 values
  auto i32Type = rewriter.getI32Type();
  auto packedMemRefType = MemRefType::get(signType.getShape(), i32Type);
  
  // Create DenseElementsAttr for packed values
  auto packedAttr = DenseElementsAttr::get(
      RankedTensorType::get(signType.getShape(), i32Type),
      llvm::ArrayRef(packedValues));
  
  // Generate unique name for packed global
  std::string packedName = baseName + "_packed";
  
  // Check if this global already exists
  if (auto existingGlobal = module.lookupSymbol<memref::GlobalOp>(packedName)) {
    // Return reference to existing global
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(existingGlobal);
    return rewriter.create<memref::GetGlobalOp>(loc, packedMemRefType, packedName);
  }
  
  // Create new global at module level
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  
  auto newGlobal = rewriter.create<memref::GlobalOp>(
      loc, packedName, rewriter.getStringAttr("private"),
      packedMemRefType, packedAttr, true, nullptr);
  
  // Create get_global op to reference it
  rewriter.setInsertionPoint(signGetGlobal);
  return rewriter.create<memref::GetGlobalOp>(loc, packedMemRefType, packedName);
}

// Handle scalar scale values - pack at compile time
static Value packScalarScale(
    PatternRewriter &rewriter, Location loc,
    Value sign, Value coe, Value rshift) {
  
  // Try to extract constant values
  auto signConst = sign.getDefiningOp<arith::ConstantOp>();
  auto coeConst = coe.getDefiningOp<arith::ConstantOp>();
  auto rshiftConst = rshift.getDefiningOp<arith::ConstantOp>();
  
  if (signConst && coeConst && rshiftConst) {
    // All are constants - pack at compile time
    auto signAttr = signConst.getValue().cast<IntegerAttr>();
    auto coeAttr = coeConst.getValue().cast<IntegerAttr>();
    auto rshiftAttr = rshiftConst.getValue().cast<IntegerAttr>();
    
    int8_t signVal = signAttr.getInt();
    uint32_t coeVal = coeAttr.getInt();
    int16_t rshiftVal = rshiftAttr.getInt();
    
    uint32_t packed = packScaleValues(signVal, coeVal, rshiftVal,
                                      kScalePackingAttr, kScaleCoeModeAttr, kExpectedAlloFixedBits);
    
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(packed));
  }
  
  // Not all constants - would need runtime packing (not implemented)
  return Value();
}

// Main function to convert scale (handles both global and scalar)
static Value convertScaleToPacked(
    PatternRewriter &rewriter, Location loc, ModuleOp module,
    Value sign, Value coe, Value rshift,
    const std::string &baseName) {
  
  // Try global conversion first
  if (auto packed = convertGlobalScaleToPacked(rewriter, loc, module, sign, coe, rshift, baseName)) {
    return packed;
  }
  
  // Try scalar packing
  if (auto packed = packScalarScale(rewriter, loc, sign, coe, rshift)) {
    return packed;
  }
  
  // Cannot pack - return null
  return Value();
}

//===----------------------------------------------------------------------===//
// Pattern: allo.qmatmul -> vivado.qmatmul
//===----------------------------------------------------------------------===//

struct QMatMulLoweringPattern : public OpRewritePattern<allo_ops::QMatMulOp> {
  using OpRewritePattern<allo_ops::QMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(allo_ops::QMatMulOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Extract all operands (preserve exact semantics)
    Value output = op.getOutput();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // Pack scale parameters at compile time
    Value xScale = convertScaleToPacked(rewriter, loc, module,
        op.getXScaleSign(), op.getXScaleCoe(), op.getXScaleRshift(), "x_scale");
    Value yScale = convertScaleToPacked(rewriter, loc, module,
        op.getYScaleSign(), op.getYScaleCoe(), op.getYScaleRshift(), "y_scale");
    Value oScale = convertScaleToPacked(rewriter, loc, module,
        op.getOScaleSign(), op.getOScaleCoe(), op.getOScaleRshift(), "o_scale");
    
    if (!xScale || !yScale || !oScale) {
      return op.emitError("Failed to pack scale parameters");
    }
    
    // Optional zero points
    Value xZero = op.getXZero();
    Value yZero = op.getYZero();
    Value oZero = op.getOZero();

    // Create vivado.qmatmul op with packed scales
    rewriter.replaceOpWithNewOp<vivado_ops::QMatMulOp>(
        op, output, lhs, rhs,
        xScale, yScale, oScale,
        xZero, yZero, oZero,
        // Backend-specific attributes with defaults from PYNQConfig
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileM),  // tile_m
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileN),  // tile_n
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileK),  // tile_k
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids (optional)
        rewriter.getBoolAttr(false),     // enable_bias
        rewriter.getBoolAttr(false),     // transpose_weight
        nullptr,                         // hls_pragmas (optional)
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline"), // requant_mode
        rewriter.getStringAttr(kScalePackingAttr), // scale_packing
        rewriter.getStringAttr(kScaleCoeModeAttr)   // scale_coe_mode
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
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    
    // Extract all operands
    Value output = op.getOutput();
    Value input = op.getInput();
    Value weight = op.getWeight();
    
    // Pack scale parameters at compile time
    Value fscl = convertScaleToPacked(rewriter, loc, module,
        op.getFsclSign(), op.getFsclCoe(), op.getFsclRshift(), "fscl");
    Value iscl = convertScaleToPacked(rewriter, loc, module,
        op.getIsclSign(), op.getIsclCoe(), op.getIsclRshift(), "iscl");
    Value oscl = convertScaleToPacked(rewriter, loc, module,
        op.getOsclSign(), op.getOsclCoe(), op.getOsclRshift(), "oscl");
    Value wscl = convertScaleToPacked(rewriter, loc, module,
        op.getWsclSign(), op.getWsclCoe(), op.getWsclRshift(), "wscl");
    
    // Optional bias scale
    Value bscl;
    if (op.getBsclSign()) {
      bscl = convertScaleToPacked(rewriter, loc, module,
          op.getBsclSign(), op.getBsclCoe(), op.getBsclRshift(), "bscl");
    }
    
    if (!fscl || !iscl || !oscl || !wscl) {
      return op.emitError("Failed to pack scale parameters");
    }
    
    // Optional parameters
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    Value bias = op.getBias();

    // Create vivado.qlinear op with packed scales
    rewriter.replaceOpWithNewOp<vivado_ops::QLinearOp>(
        op, output, input, weight,
        fscl, iscl, oscl, wscl, bscl,
        inputZero, outputZero, bias,
        // Backend-specific attributes from PYNQConfig
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileM),  // tile_m
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileN),  // tile_n
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileK),  // tile_k
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids (optional)
        rewriter.getI32IntegerAttr(pynq::BiasConfig::kDefaultBiasBufferId),   // bias_buffer_id
        rewriter.getBoolAttr(true),      // fuse_bias
        rewriter.getBoolAttr(false),     // transpose_weight
        nullptr,                         // hls_pragmas (optional)
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline"), // requant_mode
        rewriter.getStringAttr(kScalePackingAttr), // scale_packing
        rewriter.getStringAttr(kScaleCoeModeAttr)   // scale_coe_mode
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
    
    // Pack x scale at compile time
    Value xScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getXScaleSign(), op.getXScaleCoe(), op.getXScaleRshift(), "x_scale"
      );
    if (!xScale) return failure();
    
    // Pack y scale at compile time
    Value yScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getYScaleSign(), op.getYScaleCoe(), op.getYScaleRshift(), "y_scale"
      );
    if (!yScale) return failure();
    
    // Pack o scale at compile time
    Value oScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getOScaleSign(), op.getOScaleCoe(), op.getOScaleRshift(), "o_scale"
      );
    if (!oScale) return failure();
    
    Value xZero = op.getXZero();
    Value yZero = op.getYZero();
    Value oZero = op.getOZero();

    // Create vivado.qadd op with vectorization hints
    rewriter.replaceOpWithNewOp<vivado_ops::QAddOp>(
        op, output, lhs, rhs,
        xScale, yScale, oScale,
        xZero, yZero, oZero,
        // Backend-specific attributes
        rewriter.getBoolAttr(false),  // fuse_into_producer
        rewriter.getBoolAttr(true),    // vectorize
        rewriter.getStringAttr(kScalePackingAttr),
        rewriter.getStringAttr(kScaleCoeModeAttr)
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
    
    // Pack input_scale at compile time
    Value inputScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getInputScaleSign(), op.getInputScaleCoe(), op.getInputScaleRshift(), "input_scale"
      );
    if (!inputScale) return failure();
    
    // Pack gelu_scale at compile time
    Value geluScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getGeluScaleSign(), op.getGeluScaleCoe(), op.getGeluScaleRshift(), "gelu_scale"
      );
    if (!geluScale) return failure();
    
    // Pack fused_scale at compile time
    Value fusedScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getFusedScaleSign(), op.getFusedScaleCoe(), op.getFusedScaleRshift(), "fused_scale"
      );
    if (!fusedScale) return failure();
    
    // Pack output_scale at compile time
    Value outputScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getOutputScaleSign(), op.getOutputScaleCoe(), op.getOutputScaleRshift(), "output_scale"
      );
    if (!outputScale) return failure();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();

    // Create vivado.int_gelu with LUT implementation and all 4 scale groups
    rewriter.replaceOpWithNewOp<vivado_ops::IntGELUOp>(
        op, output, input,
        inputScale, geluScale, fusedScale, outputScale,
        inputZero, outputZero,
        // Backend-specific attributes
        rewriter.getStringAttr("lut"),  // implementation: "lut" or "polynomial"
        rewriter.getStringAttr(kScalePackingAttr),
        rewriter.getStringAttr(kScaleCoeModeAttr)
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
    
    // Pack input_scale at compile time
    Value inputScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getInputScaleSign(), op.getInputScaleCoe(), op.getInputScaleRshift(), "input_scale"
      );
    if (!inputScale) return failure();
    
    // Pack softmax_scale at compile time
    Value softmaxScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getSoftmaxScaleSign(), op.getSoftmaxScaleCoe(), op.getSoftmaxScaleRshift(), "softmax_scale"
      );
    if (!softmaxScale) return failure();

    // Pack fused_scale at compile time
    Value fusedScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getFusedScaleSign(), op.getFusedScaleCoe(), op.getFusedScaleRshift(), "fused_scale"
      );
    if (!fusedScale) return failure();

    // Pack output_scale at compile time
    Value outputScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getOutputScaleSign(), op.getOutputScaleCoe(), op.getOutputScaleRshift(),   "output_scale"
      );
    if (!outputScale) return failure();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    
    // TODO：add axis parameter
    int64_t axis = 1; // op.getAxis();

    // Create vivado.int_softmax with LUT implementation
    rewriter.replaceOpWithNewOp<vivado_ops::IntSoftmaxOp>(
        op, output, input,
        inputScale, softmaxScale, outputScale, fusedScale,
        inputZero, outputZero,
        rewriter.getI64IntegerAttr(axis),
        // Backend-specific attributes
        rewriter.getStringAttr("lut"),  // implementation
        rewriter.getStringAttr(kScalePackingAttr),
        rewriter.getStringAttr(kScaleCoeModeAttr)
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
    
    // Pack input_scale at compile time
    Value inputScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getInputScaleSign(), op.getInputScaleCoe(), op.getInputScaleRshift(), "input_scale"
      );
    if (!inputScale) return failure();

    // Pack layernorm_scale at compile time
    Value layernormScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getLayernormScaleSign(), op.getLayernormScaleCoe(), op.getLayernormScaleRshift(), "layernorm_scale"
      );
    if (!layernormScale) return failure();
    
    // Pack bias_scale at compile time
    Value biasScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getBiasScaleSign(), op.getBiasScaleCoe(), op.getBiasScaleRshift(), "bias_scale"
      );
    if (!biasScale) return failure();

    // Pack fused_scale at compile time
    Value fusedScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getFusedScaleSign(), op.getFusedScaleCoe(), op.getFusedScaleRshift(), "fused_scale"
      );
    if (!fusedScale) return failure();

    // Pack output_scale at compile time
    Value outputScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getOutputScaleSign(), op.getOutputScaleCoe(), op.getOutputScaleRshift(), "output_scale"
      );
    if (!outputScale) return failure();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    
    // TODO: add eps parameter
    float eps = 1e-5; // op.getEps();

    // Create vivado.int_layernorm with approx rsqrt
    rewriter.replaceOpWithNewOp<vivado_ops::IntLayerNormOp>(
        op, output, input, biasInt,
        inputScale, layernormScale, biasScale, fusedScale, outputScale,
        inputZero, outputZero,
        rewriter.getF32FloatAttr(eps),
        // Backend-specific attributes
        rewriter.getStringAttr("approx"),  // rsqrt_method
        rewriter.getStringAttr(kScalePackingAttr),
        rewriter.getStringAttr(kScaleCoeModeAttr)
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
    
    // Pack fscl at compile time
    Value fscale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getFsclSign(), op.getFsclCoe(), op.getFsclRshift(), "fscl"
      );
    if (!fscale) return failure();
    
    // Pack iscl at compile time
    Value iscale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getIsclSign(), op.getIsclCoe(), op.getIsclRshift(), "iscl"
      );
    if (!iscale) return failure();
    
    // Pack oscl at compile time
    Value oscale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getOsclSign(), op.getOsclCoe(), op.getOsclRshift(), "oscl"
      );
    if (!oscale) return failure();
    
    // Pack wscl at compile time
    Value wscale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getWsclSign(), op.getWsclCoe(), op.getWsclRshift(), "wscl"
      );
    if (!wscale) return failure();
    
    // Pack bscl at compile time
    Value bscale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getBsclSign(), op.getBsclCoe(), op.getBsclRshift(), "bscl"
      );
    if (!bscale) return failure();
    
    Value inputZero = op.getInputZero();
    Value outputZero = op.getOutputZero();
    Value bias = op.getBias();
    
    auto stride = op.getStride();

    rewriter.replaceOpWithNewOp<vivado_ops::QConv2dOp>(
        op, output, input, filter,
        fscale, iscale, oscale, wscale, bscale,
        inputZero, outputZero, bias,
        rewriter.getDenseI64ArrayAttr(stride),
        // Backend-specific attributes from PYNQConfig
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileM),  // tile_h
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileN),  // tile_w
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileK),  // tile_c
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids
        rewriter.getBoolAttr(true),      // fuse_bias
        nullptr,                         // hls_pragmas
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline"), // requant_mode
        rewriter.getStringAttr(kScalePackingAttr),
        rewriter.getStringAttr(kScaleCoeModeAttr)
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
    
    // Pack x scale at compile time
    Value xScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getXScaleSign(), op.getXScaleCoe(), op.getXScaleRshift(), "x_scale"
      );
    if (!xScale) return failure();
    
    // Pack y scale at compile time
    Value yScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getYScaleSign(), op.getYScaleCoe(), op.getYScaleRshift(), "y_scale"
      );
    if (!yScale) return failure();
    
    // Pack o scale at compile time
    Value oScale = convertScaleToPacked(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(),
        op.getOScaleSign(), op.getOScaleCoe(), op.getOScaleRshift(), "o_scale"
      );
    if (!oScale) return failure();
    
    Value xZero = op.getXZero();
    Value yZero = op.getYZero();
    Value oZero = op.getOZero();

    rewriter.replaceOpWithNewOp<vivado_ops::QMatMulIsqrtDOp>(
        op, output, lhs, rhs,
        xScale, yScale, oScale,
        xZero, yZero, oZero,
        // Backend-specific attributes from PYNQConfig
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileM),  // tile_m
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileN),  // tile_n
        rewriter.getI32IntegerAttr(pynq::TileConfig::kDefaultTileK),  // tile_k
        rewriter.getStringAttr("auto"),  // buffer_strategy
        nullptr,                         // buffer_ids
        rewriter.getBoolAttr(false),     // enable_bias
        rewriter.getBoolAttr(false),     // transpose_weight
        nullptr,                         // hls_pragmas
        rewriter.getStringAttr("i32"),   // accumulator_type
        rewriter.getStringAttr("inline"), // requant_mode
        rewriter.getStringAttr(kScalePackingAttr),
        rewriter.getStringAttr(kScaleCoeModeAttr)
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

bool applyLowerAlloQuantToVivado(ModuleOp &module, MLIRContext *context) {
  // Read and validate fixed_bits from module attribute
  bool validationFailed = false;
  int actualFixedBits = getAndValidateFixedBits(module, validationFailed);
  
  // Emit error if validation failed but continue with default for debugging
  if (validationFailed) {
    module.emitError()
        << "Fixed-bits configuration mismatch detected. "
        << "Actual: " << actualFixedBits << ", Expected: " << kExpectedAlloFixedBits << ". "
        << "Please update either SCALE_FIXED_BITS in Python or kExpectedAlloFixedBits in C++.";
    // Continue with lowering to allow developers to see other errors
  }
  
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
