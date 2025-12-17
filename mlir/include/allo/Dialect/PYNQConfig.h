/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_PYNQ_CONFIG_H
#define ALLO_PYNQ_CONFIG_H

#include <cstdint>

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

/// Instruction encoding configuration
struct InstrConfig {
  /// Buffer ID field width (3 bits, range 0-7)
  static constexpr unsigned kBufferIdWidth = 3;
  
  /// Tile count field width (3 bits, represents 1-8 tiles)
  static constexpr unsigned kTileCountWidth = 3;
  
  /// Package count field width (12 bits, 0-4095)
  static constexpr unsigned kPackageCountWidth = 12;
  
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
