/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_PYNQ_CONFIG_H
#define ALLO_PYNQ_CONFIG_H

#include <cstdint>
#include <string>

namespace mlir {
namespace allo {
namespace pynq {

//===----------------------------------------------------------------------===//
// PYNQ Hardware Configuration Constants
//===----------------------------------------------------------------------===//

/// Hardware buffer configuration
struct BufferConfig {
  /// Number of on-chip buffers available
  static constexpr unsigned kNumBuffers = 8;
  
  /// Maximum buffer ID (0-7 for 8 buffers)
  static constexpr unsigned kMaxBufferId = kNumBuffers - 1;
  
  /// Default buffer capacity in bytes (64KB)
  static constexpr int64_t kDefaultCapacityBytes = 65536;
  
  /// Minimum buffer capacity in bytes
  static constexpr int64_t kMinCapacityBytes = 1;
  
  /// Maximum buffer capacity in bytes
  static constexpr int64_t kMaxCapacityBytes = 65536;
};

/// Tile configuration for matrix operations
struct TileConfig {
  /// Default tile size for M dimension (rows)
  static constexpr int32_t kDefaultTileM = 256;
  
  /// Default tile size for N dimension (columns)
  static constexpr int32_t kDefaultTileN = 256;
  
  /// Default tile size for K dimension (reduction)
  static constexpr int32_t kDefaultTileK = 256;
  
  /// Systolic array size (hardware constraint)
  static constexpr int32_t kSystolicSize = 256;
  
  /// Number of elements per tile (DATA_PER_STREAM in hardware)
  static constexpr int32_t kElementsPerTile = 65536;
  
  /// Bytes per tile for i8 elements
  static constexpr int32_t kBytesPerTileI8 = 65536;
};

/// DMA transfer configuration
struct DMAConfig {
  /// Bytes per DMA package (256 bits = 32 bytes)
  static constexpr int32_t kBytesPerPackage = 32;
  
  /// Elements per package for i8
  static constexpr int32_t kElementsPerPackageI8 = 32;
  
  /// Maximum package count (12-bit field in instruction)
  static constexpr int32_t kMaxPackageCount = 4095;
  
  /// Direction: host to device (load)
  static constexpr int32_t kDirectionLoad = 0;
  
  /// Direction: device to host (store)
  static constexpr int32_t kDirectionStore = 1;
};

/// FIFO transfer configuration for scale/bias streaming
struct FIFOConfig {
  /// Chunk granularity (in elements) for matrix-op scales
  static constexpr int32_t kMatrixScaleChunk = 32;

  /// Chunk granularity (in elements) for vector-op scales
  static constexpr int32_t kVectorScaleChunk = 32;

  /// Chunk granularity (in elements) for vector-op bias (used by LayerNorm)
  static constexpr int32_t kBiasChunk = 64;

  /// Direction code for transferring matrix scales to matrix FIFO
  static constexpr int32_t kDirectionMatrixScaleFIFO = 2;

  /// Direction code for transferring vector scales/bias to vector FIFO
  static constexpr int32_t kDirectionVectorScaleFIFO = 3;
};

/// Instruction encoding configuration
struct InstrConfig {
  /// Buffer ID field width (3 bits, range 0-7)
  static constexpr unsigned kBufferIdWidth = 3;
  
  /// Tile count field width (3 bits, represents 1-8 tiles)
  static constexpr unsigned kTileCountWidth = 3;

  /// Tile Size (32 Bytes)
  // NOTE: This is different from TileConfig kDefaultTileM which is in elements
  // TileConfig::kDefaultTileM is a Matrix which is supported by the hardware standardly
  // InstrConfig::kTileSize is the basic unit size in the instruction encoding the columns (not rows)
  static constexpr int32_t kTileSize = 32;

  /// Package count field width (12 bits, 0-4095)
  static constexpr unsigned kPackageCountWidth = 12;

  /// Package size (32 Bytes)
  static constexpr int32_t kPackageSize = 32;
  
  /// Reduce K field width (8 bits, 0-255)
  static constexpr unsigned kReduceKWidth = 8;
  
  /// Maximum tile count (1-8)
  static constexpr unsigned kMaxTileCount = 8;
  
  /// Maximum reduce K value (1-256)
  static constexpr unsigned kMaxReduceK = 256;
  
  /// Maximum head count for multi-head attention (0-7)
  static constexpr unsigned kMaxHeadCount = 7;
};

/// Bias configuration
struct BiasConfig {
  /// Default bias buffer ID (buffer 7 reserved for bias)
  static constexpr unsigned kDefaultBiasBufferId = 7;
  
  /// Bias elements per stream (BIAS_PER_STREAM in hardware)
  static constexpr int32_t kBiasPerStream = 8;
};

/// Vector operation codes
/// just as a doc
struct VectorOpCode {
  /// GELU activation
  static constexpr unsigned kGELU = 0;
  
  /// Quantized element-wise addition
  static constexpr unsigned kQAdd = 1;
  
  /// Softmax
  static constexpr unsigned kSoftmax = 2;
  
  /// Layer normalization
  static constexpr unsigned kLayerNorm = 3;
};

/// Scale mode enumeration for quantization
struct ScaleMode {
  /// Per-tensor quantization: single scale for entire tensor
  static constexpr unsigned kPerTensor = 0;
  
  /// Per-channel quantization: separate scale per channel (output dimension)
  static constexpr unsigned kPerChannel = 1;
  
  /// Per-token quantization: separate scale per token (sequence dimension)
  static constexpr unsigned kPerToken = 2;
};

/// Canonical packed-scale format configuration used by quantized lowering.
///
/// Packing format string schema:
///   "sign_bits,sign_offset,rshift_bits,rshift_offset,coe_bits,coe_offset"
///
/// Coe mode:
/// - "Tail": coefficient excludes implicit leading 1
/// - "Full": coefficient includes explicit leading 1 (or sign-fused form)
struct PackedScaleConfig {
  /// Source format before repacking.
  static constexpr const char *kSourcePackingAttr = "8,24,8,16,16,0";
  static constexpr const char *kSourceCoeModeAttr = "Tail";

  /// Target canonical format after repacking.
  static constexpr const char *kTargetPackingAttr = "0,22,6,16,16,0";
  static constexpr const char *kTargetCoeModeAttr = "Full";
};

/// Fixed-point fractional-bit configuration aligned with
/// externals/transformer_lib/example/quantize/config/hyper_config.py.
struct VectorFixedPointConfig {
  /// GELU input/output fixed-point formats: Fixed(2, 13)
  static constexpr int32_t kGELUInputFracBits = 13;
  static constexpr int32_t kGELUOutputFracBits = 13;

  /// Softmax fixed-point formats: x=Fixed(3, 12), ex=Fixed(0, 16, False)
  static constexpr int32_t kSoftmaxInputFracBits = 12;
  static constexpr int32_t kSoftmaxExpFracBits = 16;
  static constexpr int32_t kSoftmaxLogEFracBits = 12;

  /// LayerNorm fixed-point formats from hyper_config.py
  /// norm_x = Fixed(16, 0, False), norm_insqrt = Fixed(0, 16, False)
  static constexpr int32_t kLayerNormInputFracBits = 0;
  static constexpr int32_t kLayerNormInvSqrtFracBits = 16;

  /// Shortcut(QAdd) input/output fixed-point formats: Fixed(7, 8)
  static constexpr int32_t kShortcutInputFracBits = 8;
  static constexpr int32_t kShortcutOutputFracBits = 8;
};

/// Rshift adjustment configuration used by PYNQ scale-adjust pass.
struct RShiftAdjustConfig {
  /// Innate front-end bias added in float_to_fixed_point.
  static constexpr int32_t kInnateRshiftBias = 16;

  /// MatMul accumulator drop width used by matmul quant flow.
  static constexpr int32_t kMatMulDropWidth = 8;
};

/// QLinear layer type identification
struct QLinearLayerType {
  /// Query projection in multi-head attention
  static constexpr const char* kQKVGemmProjQ = "qkvgemm.proj_q";
  
  /// Key projection in multi-head attention
  static constexpr const char* kQKVGemmProjK = "qkvgemm.proj_k";
  
  /// Value projection in multi-head attention
  static constexpr const char* kQKVGemmProjV = "qkvgemm.proj_v";
  
  /// Attention output projection
  static constexpr const char* kAttnGemmProj = "attngemm.proj";
  
  /// First FC layer in feed-forward network
  static constexpr const char* kFFNFC1 = "ffn.fc1";
  
  /// Second FC layer in feed-forward network
  static constexpr const char* kFFNFC2 = "ffn.fc2";
  
  /// Classification head dense layer
  static constexpr const char* kClassifierDense = "classifier.dense";
  
  /// Unknown or unspecified layer type
  static constexpr const char* kUnknown = "unknown";
  
  /// Check if layer type is Query projection
  static bool isQKVGemmProjQ(llvm::StringRef layerType) {
    return layerType == kQKVGemmProjQ;
  }
  
  /// Check if layer type is Key projection
  static bool isQKVGemmProjK(llvm::StringRef layerType) {
    return layerType == kQKVGemmProjK;
  }
  
  /// Check if layer type is Value projection
  static bool isQKVGemmProjV(llvm::StringRef layerType) {
    return layerType == kQKVGemmProjV;
  }
  
  /// Check if layer type is any Q/K/V projection
  static bool isQKVGemmProj(llvm::StringRef layerType) {
    return isQKVGemmProjQ(layerType) || isQKVGemmProjK(layerType) || 
           isQKVGemmProjV(layerType);
  }
  
  /// Check if layer type is Attention output projection
  static bool isAttnGemmProj(llvm::StringRef layerType) {
    return layerType == kAttnGemmProj;
  }
  
  /// Check if layer type is FFN first FC layer
  static bool isFFNFC1(llvm::StringRef layerType) {
    return layerType == kFFNFC1;
  }
  
  /// Check if layer type is FFN second FC layer
  static bool isFFNFC2(llvm::StringRef layerType) {
    return layerType == kFFNFC2;
  }
  
  /// Check if layer type is any FFN layer
  static bool isFFNLayer(llvm::StringRef layerType) {
    return isFFNFC1(layerType) || isFFNFC2(layerType);
  }
  
  /// Check if layer type is Classifier dense layer
  static bool isClassifierDense(llvm::StringRef layerType) {
    return layerType == kClassifierDense;
  }
  
  /// Check if layer type is Unknown
  static bool isUnknown(llvm::StringRef layerType) {
    return layerType == kUnknown;
  }
  
  /// Check if layer type is in multi-head attention block (Q/K/V or attention proj)
  static bool isAttentionLayer(llvm::StringRef layerType) {
    return isQKVGemmProj(layerType) || isAttnGemmProj(layerType);
  }
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Calculate number of DMA packages needed for a given size in bytes
inline int64_t calculatePackageCount(int64_t sizeBytes) {
  return (sizeBytes + DMAConfig::kBytesPerPackage - 1) / 
         DMAConfig::kBytesPerPackage;
}

/// Calculate number of tiles needed for a dimension
inline int64_t calculateTileCount(int64_t dimension, int32_t tileSize) {
  return (dimension + tileSize - 1) / tileSize;
}

/// Check if dimension is divisible by tile size
inline bool isDivisibleByTile(int64_t dimension, int32_t tileSize) {
  return (dimension % tileSize) == 0;
}

/// Calculate buffer capacity needed for a tile
inline int64_t calculateTileCapacity(int32_t tileM, int32_t tileN, 
                                      unsigned elementSizeBytes) {
  return static_cast<int64_t>(tileM) * tileN * elementSizeBytes;
}

/// Validate tile size fits in buffer
inline bool validateTileSize(int32_t tileM, int32_t tileN, 
                              unsigned elementSizeBytes,
                              int64_t bufferCapacity = BufferConfig::kDefaultCapacityBytes) {
  return calculateTileCapacity(tileM, tileN, elementSizeBytes) <= bufferCapacity;
}

} // namespace pynq
} // namespace allo
} // namespace mlir

#endif // ALLO_PYNQ_CONFIG_H
