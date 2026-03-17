/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// RepackVivadoScales Pass
// This pass converts packed scale parameters from one format to another.
//
// Packing Format: "sign_bits,sign_offset,rshift_bits,rshift_offset,coe_bits,coe_offset"
//
// Design Principles:
// 1. When sign_bits > 0: Supports "Full" and "Tail" coe modes
//    - Full: coe includes leading 1 (the 0.5 fractional bit)
//    - Tail: coe excludes leading 1 (only fractional part)
// 2. Sign bit encoding:
//    - sign_bits > 1: Can directly represent +1 and -1
//    - sign_bits == 1: 0=positive, 1=negative
// 3. When sign_bits == 0: Sign is fused into coe
//    - coe is always Full mode with two's complement encoding
//    - MSB of coe acts as sign bit
// 4. Unpacked representation: sign/coe/rshift use sufficient width, coe always Full
// 5. Conversion: Based on source pack format, not hardcoded constants
//
// Example Source: "8,24,8,16,16,0" (Tail mode, separate sign)
//   Layout: sign[31:24] | rshift[23:16] | coe[15:0]
// Example Target: "0,22,6,16,16,0" (Full mode, sign-fused two's complement)
//   Layout: rshift[27:22] | coe[21:0]
//===----------------------------------------------------------------------===/

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/PYNQConfig.h"
#include "allo/Dialect/VivadoOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace allo;
namespace vivado_ops = mlir::allo::vivado;

namespace mlir {
namespace allo {

//===----------------------------------------------------------------------===//
// Scale Repacking Helper Functions
//===----------------------------------------------------------------------===//

// Parse packing configuration string
struct PackingConfig {
  int sign_bits;
  int sign_offset;
  int rshift_bits;
  int rshift_offset;
  int coe_bits;
  int coe_offset;
  std::string coe_mode;
  
  static PackingConfig parse(const std::string &attr) {
    PackingConfig config;
    std::vector<int> values;
    std::string str = attr;
    size_t pos = 0;
    
    while ((pos = str.find(',')) != std::string::npos) {
      values.push_back(std::stoi(str.substr(0, pos)));
      str.erase(0, pos + 1);
    }
    values.push_back(std::stoi(str));
    
    if (values.size() != 6) {
      return {8, 24, 8, 16, 16, 0};  // Default
    }
    
    config.sign_bits = values[0];
    config.sign_offset = values[1];
    config.rshift_bits = values[2];
    config.rshift_offset = values[3];
    config.coe_bits = values[4];
    config.coe_offset = values[5];

    config.coe_mode = "Full"; // Default
    
    return config;
  }
};

// Unpack a i32 value according to source format
// Returns Full representation: coe always includes leading 1, sign is separated
struct UnpackedScale {
  int8_t sign;      // +1 or -1 (extracted from sign bits or coe MSB)
  int16_t rshift;   // shift amount
  uint32_t coe;     // coefficient (always Full mode: includes leading 1)
};

// Unpack a i32 value according to source format
// Converts to normalized Full representation for internal processing
static UnpackedScale unpackScale(uint32_t packed, 
                                  const PackingConfig &config) {
  UnpackedScale result;
  std::string coe_mode = config.coe_mode;
  
  // Extract rshift (always present)
  uint32_t rshift_mask = (1u << config.rshift_bits) - 1;
  result.rshift = (packed >> config.rshift_offset) & rshift_mask;
  
  // Extract coe bits
  uint32_t coe_mask = (1u << config.coe_bits) - 1;
  uint32_t coe_raw = (packed >> config.coe_offset) & coe_mask;

  if (config.sign_bits > 0) {
    // Case 1: Separate sign bits
    uint32_t sign_mask = (1u << config.sign_bits) - 1;
    uint32_t sign_val = (packed >> config.sign_offset) & sign_mask;
    
    if (config.sign_bits == 1) {
      result.sign = (sign_val == 0) ? 1 : -1;  // 0=positive, 1=negative
    } else {
      // sign_bits > 1: can directly represent +1 and -1
      result.sign = (sign_val == 1) ? 1 : -1;
    }
    
    // Convert coe to Full mode
    if (coe_mode == "Tail") {
      // Tail: add leading 1 to get Full representation
      result.coe = (1u << config.coe_bits) | coe_raw;
    } else { // "Full"
      result.coe = coe_raw;
    }
  } else {
    // Case 2: Sign fused into coe (two's complement)
    // coe is always Full mode here
    bool is_negative = (coe_raw >> (config.coe_bits - 1)) & 0x1;
    result.sign = is_negative ? -1 : 1;
    
    if (is_negative) {
      // Convert from two's complement to positive magnitude
      // Two's complement: -x = ~x + 1
      // To get magnitude: x = ~(-x) + 1 = ~coe_raw + 1
      result.coe = (~coe_raw + 1) & coe_mask;
    } else {
      result.coe = coe_raw;
    }
  }
  
  return result;
}

// Repack scale values into target format
// Input: unpacked scale (coe in Full mode, sign separated)
// Handles:
// 1. Bit width adjustment based on source/target Full representation
// 2. Full/Tail mode conversion for target
// 3. Sign fusion (two's complement) if target sign_bits == 0
// 4. rshift correction for bit width changes
static uint32_t repackScale(const UnpackedScale &unpacked, 
                            const PackingConfig &srcConfig,
                            const PackingConfig &tgtConfig) {
  std::string src_mode = srcConfig.coe_mode;
  std::string tgt_mode = tgtConfig.coe_mode;
  // Unpacked coe is always Full, calculate its actual bit width
  // For separate sign: full width = coe_bits (Tail) or coe_bits (Full)
  // But unpacked is always Full, so width = coe_bits + 1 (if Tail) or coe_bits (if Full)
  int actual_src_full_bits;
  if (srcConfig.sign_bits > 0) {
    actual_src_full_bits = (src_mode == "Tail") ? (srcConfig.coe_bits + 1) : srcConfig.coe_bits;
  } else {
    // Sign-fused: MSB is sign, so effective magnitude bits = coe_bits - 1
    actual_src_full_bits = srcConfig.coe_bits - 1;
  }
  
  // Calculate target Full representation bit width
  int actual_tgt_full_bits;
  if (tgtConfig.sign_bits > 0) {
    actual_tgt_full_bits = (tgt_mode == "Tail") ? (tgtConfig.coe_bits + 1) : tgtConfig.coe_bits;
  } else {
    actual_tgt_full_bits = tgtConfig.coe_bits - 1;
  }
  
  // Step 1: Adjust bit width (scale coe value)
  int bit_diff = actual_tgt_full_bits - actual_src_full_bits;
  uint32_t coe_adjusted = unpacked.coe;
  int16_t rshift_adjusted = unpacked.rshift;
  
  if (bit_diff > 0) {
    // Target has more bits: left shift (increase precision)
    coe_adjusted = coe_adjusted << bit_diff;
  } else if (bit_diff < 0) {
    // Target has fewer bits: right shift (reduce precision)
    coe_adjusted = coe_adjusted >> (-bit_diff);
  }
  rshift_adjusted += bit_diff;  // Compensate in rshift
  
  // Step 2: Convert to target coe representation
  uint32_t coe_to_pack;
  if (tgtConfig.sign_bits > 0) {
    // Target has separate sign: convert Full to Full/Tail
    if (tgt_mode == "Tail") {
      // Remove leading 1 for Tail mode
      uint32_t leading_bit_pos = actual_tgt_full_bits - 1;
      coe_to_pack = coe_adjusted & ((1u << leading_bit_pos) - 1);
    } else { // "Full"
      coe_to_pack = coe_adjusted;
    }
  } else {
    // Target has sign fused: convert to two's complement
    if (unpacked.sign < 0) {
      // Negate using two's complement: -x = ~x + 1
      uint32_t mask = (1u << tgtConfig.coe_bits) - 1;
      coe_to_pack = (~(coe_adjusted - 1)) & mask;
    } else {
      coe_to_pack = coe_adjusted;
    }
  }
  
  // Step 3: Pack into target format
  uint32_t packed = 0;
  
  // Pack sign (if separate)
  if (tgtConfig.sign_bits > 0) {
    uint32_t sign_val;
    if (tgtConfig.sign_bits == 1) {
      sign_val = (unpacked.sign < 0) ? 1 : 0;
    } else {
      sign_val = (unpacked.sign < 0) ? static_cast<uint32_t>(-1) : 1;
    }
    uint32_t sign_mask = (1u << tgtConfig.sign_bits) - 1;
    packed |= ((sign_val & sign_mask) << tgtConfig.sign_offset);
  }
  
  // Pack rshift
  uint32_t rshift_mask = (1u << tgtConfig.rshift_bits) - 1;
  packed |= ((rshift_adjusted & rshift_mask) << tgtConfig.rshift_offset);
  
  // Pack coe
  uint32_t coe_mask = (1u << tgtConfig.coe_bits) - 1;
  packed |= ((coe_to_pack & coe_mask) << tgtConfig.coe_offset);
  
  return packed;
}

// Convert a memref::GlobalOp with packed i32 scales (in-place modification)
static bool convertGlobalPackedScale(
    PatternRewriter &rewriter, 
    memref::GlobalOp globalOp,
    const PackingConfig &srcConfig,
    const PackingConfig &tgtConfig) {
  
  auto initialValue = globalOp.getInitialValue();
  if (!initialValue) {
    return false;
  }
  
  auto denseAttr = initialValue->dyn_cast<DenseElementsAttr>();
  if (!denseAttr) {
    return false;
  }
  
  auto memrefType = globalOp.getType().cast<MemRefType>();
  if (!memrefType.getElementType().isInteger(32)) {
    return false;  // Only handle i32 packed scales
  }
  
  // Extract and convert packed values
  SmallVector<uint32_t> newValues;
  for (auto val : denseAttr.getValues<uint32_t>()) {
    auto unpacked = unpackScale(val, srcConfig);
    uint32_t repacked = repackScale(unpacked, srcConfig, tgtConfig);
    newValues.push_back(repacked);
  }
  
  // Create new DenseElementsAttr
  auto newAttr = DenseElementsAttr::get(
      RankedTensorType::get(memrefType.getShape(), rewriter.getI32Type()),
      llvm::ArrayRef(newValues));
  
  // Update the global op in-place
  globalOp.setInitialValueAttr(newAttr);
  
  return true;
}

// Convert a scalar constant i32 packed scale
static Value convertScalarPackedScale(
    PatternRewriter &rewriter,
    Location loc,
    Value packedScale,
    const PackingConfig &srcConfig,
    const PackingConfig &tgtConfig) {
  
  auto constOp = packedScale.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return Value();  // Not a constant
  }
  
  auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>();
  if (!intAttr) {
    return Value();
  }
  
  uint32_t packed = intAttr.getInt();
  auto unpacked = unpackScale(packed, srcConfig);
  uint32_t repacked = repackScale(unpacked, srcConfig, tgtConfig);
  
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(repacked));
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns for Vivado Ops
//===----------------------------------------------------------------------===//

// Base template for rewriting Vivado ops with packed scales
template<typename VivadoOp>
struct RepackVivadoOpPattern : public OpRewritePattern<VivadoOp> {
  using OpRewritePattern<VivadoOp>::OpRewritePattern;
  
  PackingConfig srcConfig;
  PackingConfig tgtConfig;
  std::string src_mode;
  std::string tgt_mode;
  
  RepackVivadoOpPattern(MLIRContext *context)
      : OpRewritePattern<VivadoOp>(context),
        srcConfig(PackingConfig::parse(pynq::PackedScaleConfig::kSourcePackingAttr)),
        tgtConfig(PackingConfig::parse(pynq::PackedScaleConfig::kTargetPackingAttr)),
        src_mode(pynq::PackedScaleConfig::kSourceCoeModeAttr),
        tgt_mode(pynq::PackedScaleConfig::kTargetCoeModeAttr) {}
  
  // Helper to convert scale operands
  SmallVector<Value> convertScaleOperands(
      PatternRewriter &rewriter,
      Location loc,
      ValueRange scales) const {
    SmallVector<Value> convertedScales;
    
    for (Value scale : scales) {
      if (auto converted = convertScalarPackedScale(
              rewriter, loc, scale, srcConfig, tgtConfig)) {
        convertedScales.push_back(converted);
      } else {
        convertedScales.push_back(scale);  // Keep original if can't convert
        // Note: this scale may be from a global, which will been converted in-place
      }
    }
    
    return convertedScales;
  }
};

// Pattern for vivado.qmatmul
struct RepackQMatMulPattern : public RepackVivadoOpPattern<vivado_ops::QMatMulOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::QMatMulOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(),
      {op.getXScale(), op.getYScale(), op.getFusedScale(), op.getOScale(), op.getOScaleInv()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::QMatMulOp>(
        op,
        op.getOutput(), op.getLhs(), op.getRhs(),
      convertedScales[0], convertedScales[1], convertedScales[2], convertedScales[3], convertedScales[4],
        op.getXZero(), op.getYZero(), op.getOZero(),
        op.getTileMAttr(), op.getTileNAttr(), op.getTileKAttr(),
        op.getBufferStrategyAttr(), op.getBufferIdsAttr(),
        op.getEnableBiasAttr(), op.getTransposeWeightAttr(),
        op.getHlsPragmasAttr(), op.getAccumulatorTypeAttr(),
        op.getRequantModeAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
      op.getIsTransposedAttr(),
      op.getRhsLayoutAttr(),
      op.getReduceDimAttr(),
      op.getLayerTypeAttr()
    );
    
    return success();
  }
};

// Similar patterns for other ops...
struct RepackQLinearPattern : public RepackVivadoOpPattern<vivado_ops::QLinearOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::QLinearOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    SmallVector<Value> scales = {op.getFscl(), op.getIscl(), op.getOscl(), op.getOsclInv(), op.getWscl()};
    if (op.getBscl()) scales.push_back(op.getBscl());
    
    auto convertedScales = convertScaleOperands(rewriter, op.getLoc(), scales);
    
    rewriter.replaceOpWithNewOp<vivado_ops::QLinearOp>(
        op,
        op.getOutput(), op.getInput(), op.getWeight(),
        convertedScales[0], convertedScales[1], convertedScales[2], convertedScales[3], convertedScales[4],
        convertedScales.size() > 5 ? convertedScales[5] : Value(),
        op.getInputZero(), op.getOutputZero(), op.getBias(),
        op.getTileMAttr(), op.getTileNAttr(), op.getTileKAttr(),
        op.getBufferStrategyAttr(), op.getBufferIdsAttr(),
        op.getBiasBufferIdAttr(), op.getFuseBiasAttr(),
        op.getTransposeWeightAttr(), op.getHlsPragmasAttr(),
        op.getAccumulatorTypeAttr(), op.getRequantModeAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr(),
        op.getLayerTypeAttr() 
    );
    
    return success();
  }
};

// Pattern for vivado.qadd
struct RepackQAddPattern : public RepackVivadoOpPattern<vivado_ops::QAddOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::QAddOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(),
      {op.getXScale(), op.getYScale(), op.getOScale(), op.getOScaleInv()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::QAddOp>(
        op,
        op.getOutput(), op.getLhs(), op.getRhs(),
        convertedScales[0], convertedScales[1], convertedScales[2], convertedScales[3],
        op.getXZero(), op.getYZero(), op.getOZero(),
        op.getFuseIntoProducerAttr(), op.getVectorizeAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    return success();
  }
};

// Pattern for vivado.int_gelu
struct RepackIntGELUPattern : public RepackVivadoOpPattern<vivado_ops::IntGELUOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::IntGELUOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(),
        {op.getIscl(), op.getGscl(), op.getFscl(), op.getOscl(), op.getOsclInv()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::IntGELUOp>(
        op,
        op.getOutput(), op.getInput(),
        convertedScales[0], convertedScales[1], convertedScales[2], convertedScales[3], convertedScales[4],
        op.getInputZero(), op.getOutputZero(),
        op.getImplementationAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    return success();
  }
};

// Pattern for vivado.int_softmax
struct RepackIntSoftmaxPattern : public RepackVivadoOpPattern<vivado_ops::IntSoftmaxOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::IntSoftmaxOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(),
        {op.getIscl(), op.getSscl(), op.getOscl(), op.getOsclInv(), op.getFscl()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::IntSoftmaxOp>(
        op,
        op.getOutput(), op.getInput(),
        convertedScales[0], convertedScales[1], convertedScales[2], convertedScales[3], convertedScales[4],
        op.getInputZero(), op.getOutputZero(),
        op.getAxisAttr(), op.getImplementationAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    return success();
  }
};

// Pattern for vivado.int_layernorm
struct RepackIntLayerNormPattern : public RepackVivadoOpPattern<vivado_ops::IntLayerNormOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::IntLayerNormOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(),
        {op.getIscl(), op.getLscl(), op.getBscl(), 
         op.getFscl(), op.getOscl(), op.getOsclInv()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::IntLayerNormOp>(
        op,
        op.getOutput(), op.getInput(), op.getBiasInt(),
        convertedScales[0], convertedScales[1], convertedScales[2], 
        convertedScales[3], convertedScales[4], convertedScales[5],
        op.getInputZero(), op.getOutputZero(),
        op.getEpsAttr(), op.getRsqrtMethodAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    return success();
  }
};

// Pattern for vivado.qconv2d
struct RepackQConv2dPattern : public RepackVivadoOpPattern<vivado_ops::QConv2dOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::QConv2dOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(),
        {op.getFscl(), op.getIscl(), op.getOscl(), op.getOsclInv(), op.getWscl(), op.getBscl()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::QConv2dOp>(
        op,
        op.getOutput(), op.getInput(), op.getFilter(),
        convertedScales[0], convertedScales[1], convertedScales[2], 
        convertedScales[3], convertedScales[4], convertedScales[5],
        op.getInputZero(), op.getOutputZero(), op.getBias(),
        op.getStrideAttr(),
        op.getTileHAttr(), op.getTileWAttr(), op.getTileCAttr(),
        op.getBufferStrategyAttr(), op.getBufferIdsAttr(),
        op.getEnableBiasAttr(), op.getHlsPragmasAttr(),
        op.getAccumulatorTypeAttr(), op.getRequantModeAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    return success();
  }
};

// Pattern for vivado.qmatmul_isqrtd
struct RepackQMatMulIsqrtDPattern : public RepackVivadoOpPattern<vivado_ops::QMatMulIsqrtDOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::QMatMulIsqrtDOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(),
        {op.getXScale(), op.getYScale(), op.getFusedScale(), op.getOScale(), op.getOScaleInv()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::QMatMulIsqrtDOp>(
        op,
        op.getOutput(), op.getLhs(), op.getRhs(),
      convertedScales[0], convertedScales[1], convertedScales[2], convertedScales[3], convertedScales[4],
        op.getXZero(), op.getYZero(), op.getOZero(),
        op.getTileMAttr(), op.getTileNAttr(), op.getTileKAttr(),
        op.getBufferStrategyAttr(), op.getBufferIdsAttr(),
        op.getEnableBiasAttr(), op.getTransposeWeightAttr(),
        op.getHlsPragmasAttr(), op.getAccumulatorTypeAttr(),
        op.getRequantModeAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
      op.getIsTransposedAttr(),
      op.getRhsLayoutAttr(),
      op.getReduceDimAttr(),
      op.getLayerTypeAttr()
    );
    
    return success();
  }
};

// Pattern for vivado.quant
// Note: Although global scales are repacked in-place during the first pass,
// we still need this pattern to update the scale_packing and scale_coe_mode attributes.
struct RepackQuantPattern : public RepackVivadoOpPattern<vivado_ops::QuantOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::QuantOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(), {op.getScale()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::QuantOp>(
        op,
        op.getOutput(), op.getInput(),
        convertedScales[0], op.getZero(),
        op.getQuantModeAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    return success();
  }
};

// Pattern for vivado.dequant
// Note: Although global scales are repacked in-place during the first pass,
// we still need this pattern to update the scale_packing and scale_coe_mode attributes.
struct RepackDequantPattern : public RepackVivadoOpPattern<vivado_ops::DequantOp> {
  using RepackVivadoOpPattern::RepackVivadoOpPattern;
  
  LogicalResult matchAndRewrite(vivado_ops::DequantOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if already in target format - avoid infinite loop
    if (op.getScalePacking() == pynq::PackedScaleConfig::kTargetPackingAttr && 
        op.getScaleCoeMode() == pynq::PackedScaleConfig::kTargetCoeModeAttr) {
      return failure();  // Already converted, skip
    }
    
    auto convertedScales = convertScaleOperands(
        rewriter, op.getLoc(), {op.getScale()});
    
    rewriter.replaceOpWithNewOp<vivado_ops::DequantOp>(
        op,
        op.getOutput(), op.getInput(),
        convertedScales[0], op.getZero(),
        op.getQuantModeAttr(),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetPackingAttr),
        rewriter.getStringAttr(pynq::PackedScaleConfig::kTargetCoeModeAttr),
        op.getTransposeModeAttr(),
        op.getIsTransposedAttr()
    );
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Apply Function (used by runOnOperation)
//===----------------------------------------------------------------------===//

bool applyRepackVivadoScales(ModuleOp &module, MLIRContext *context) {
  auto srcConfig = PackingConfig::parse(pynq::PackedScaleConfig::kSourcePackingAttr);
  auto tgtConfig = PackingConfig::parse(pynq::PackedScaleConfig::kTargetPackingAttr);
  srcConfig.coe_mode = pynq::PackedScaleConfig::kSourceCoeModeAttr;
  tgtConfig.coe_mode = pynq::PackedScaleConfig::kTargetCoeModeAttr;
  
  // First pass: convert all memref::GlobalOp with packed scales (in-place modification)
  module.walk([&](memref::GlobalOp globalOp) {
    std::string name = globalOp.getSymName().str();
    // Check if this is a packed scale global (contains "_packed" suffix)
    if (name.find("_packed") != std::string::npos) {
      PatternRewriter rewriter(context);
      convertGlobalPackedScale(rewriter, globalOp, srcConfig, tgtConfig);
    }
  });
  
  // Second pass: rewrite Vivado ops with converted scales
  // Note: MLIR pattern infrastructure requires creating new ops, not in-place modification
  RewritePatternSet patterns(context);
  patterns.add<RepackQMatMulPattern>(context);
  patterns.add<RepackQLinearPattern>(context);
  patterns.add<RepackQAddPattern>(context);
  patterns.add<RepackIntGELUPattern>(context);
  patterns.add<RepackIntSoftmaxPattern>(context);
  patterns.add<RepackIntLayerNormPattern>(context);
  patterns.add<RepackQConv2dPattern>(context);
  patterns.add<RepackQMatMulIsqrtDPattern>(context);
  patterns.add<RepackQuantPattern>(context);
  patterns.add<RepackDequantPattern>(context);
  
  return !failed(applyPatternsAndFoldGreedily(module, std::move(patterns)));
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RepackVivadoScalesPass
    : public PassWrapper<RepackVivadoScalesPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RepackVivadoScalesPass)

  StringRef getArgument() const final { return "repack-vivado-scales"; }
  
  StringRef getDescription() const final {
    return "Repack Vivado quantized op scale parameters from one format to another";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    
    if (!applyRepackVivadoScales(module, context)) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createRepackVivadoScalesPass() {
  return std::make_unique<RepackVivadoScalesPass>();
}

} // namespace allo
} // namespace mlir
