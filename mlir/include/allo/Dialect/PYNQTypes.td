/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * PYNQ Hardware Buffer Types
 * 
 * This file defines the type system for PYNQ on-chip buffers. The buffer type
 * supports both virtual (unallocated) and physical (allocated) representations,
 * enabling a two-phase compilation flow:
 * 
 * Phase 1 (Lowering): Generate virtual buffers with bufferId = kUnallocated
 * Phase 2 (Allocation): Assign concrete buffer IDs via allocation pass
 * 
 * Reference: PYNQ_api/include/instr.h
 *   - Buffer IDs: 3-bit encoding (0-7)
 *   - Tile organization: 32 elements per tile
 *   - DMA transfers: sized by total element count
 * 
 * Design Principles:
 *   - Buffer is a storage container (like a register in register allocation)
 *   - Shape/transpose are usage attributes, not buffer properties
 *   - Same buffer can hold different shaped data at different times
 */

#ifndef PYNQ_TYPES
#define PYNQ_TYPES

include "allo/Dialect/PYNQDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// PYNQ Buffer Type
//===----------------------------------------------------------------------===//

def PYNQ_BufferType : PYNQ_Type<"Buffer"> {
  let summary = "On-chip buffer type (virtual or physical)";
  let description = [{
    Represents an on-chip buffer in the PYNQ accelerator. This type can be
    either virtual (before allocation) or physical (after allocation).
    
    Parameters:
    1. element_type: The data type stored (i8, i32, etc.)
    2. buffer_id: Hardware buffer ID, or kUnallocated (255) for virtual buffers
    3. capacity_bytes: Buffer capacity in bytes
    
    Virtual vs Physical:
    - Virtual: `!pynq.buffer<i8, ?, 4096>` - bufferId not yet assigned
    - Physical: `!pynq.buffer<i8, 0, 4096>` - bufferId = 0 assigned
    
    The `?` in the textual format represents kUnallocated (255).
    
    Compilation Flow:
    ```
    // Stage 1: Lowering generates virtual buffers
    %buf = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
    
    // Stage 2: HoistBufferAllocPass moves allocs to function entry
    // Stage 3: BufferAllocationPass assigns IDs and merges buffers
    %buf = pynq.buffer_alloc : !pynq.buffer<i8, 0, 4096>
    ```
    
    Examples:
      !pynq.buffer<i8, ?, 1024>   // Virtual buffer, unallocated
      !pynq.buffer<i8, 0, 1024>   // Physical buffer 0
      !pynq.buffer<i32, 2, 4096>  // Physical buffer 2, i32 elements
    
    Note: Shape and transpose information are NOT part of the buffer type.
    They are usage-time attributes specified on operations that use the buffer,
    since the same buffer can hold different shaped data at different times.
  }];
  
  let mnemonic = "buffer";
  
  let parameters = (ins
    "Type":$elementType,          // Element type (i8, i32, etc.)
    "unsigned":$bufferId,         // Hardware buffer ID (0-7), or kUnallocated
    "int64_t":$capacityBytes      // Buffer capacity in bytes
  );
  
  // Custom assembly format to print '?' for unallocated buffers
  let hasCustomAssemblyFormat = 1;
  
  let genVerifyDecl = 1;
  
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Static Constants
    //===------------------------------------------------------------------===//
    
    /// Special value indicating buffer ID not yet allocated
    /// Using 255 as it's outside valid range [0, 7] but fits in unsigned
    static constexpr unsigned kUnallocated = 255;
    
    /// Maximum valid buffer ID (default: 7 for 8 buffers)
    static constexpr unsigned kMaxBufferId = 7;
    
    /// Maximum buffer capacity in bytes (default: 64KB)
    static constexpr int64_t kMaxCapacityBytes = 65536;
    
    /// Minimum buffer capacity in bytes
    static constexpr int64_t kMinCapacityBytes = 1;
    
    //===------------------------------------------------------------------===//
    // Allocation State Queries
    //===------------------------------------------------------------------===//
    
    /// Check if this buffer has been allocated a physical ID
    bool isAllocated() const {
      return getBufferId() != kUnallocated;
    }
    
    /// Check if this is a virtual (unallocated) buffer
    bool isVirtual() const {
      return getBufferId() == kUnallocated;
    }
    
    //===------------------------------------------------------------------===//
    // Verification Helper Methods
    //===------------------------------------------------------------------===//
    
    /// Check if the buffer ID is within valid range (for allocated buffers)
    bool hasValidBufferId() const {
      return isVirtual() || getBufferId() <= kMaxBufferId;
    }
    
    /// Check if the buffer ID is within a custom range [0, maxId]
    bool hasValidBufferId(unsigned maxId) const {
      return isVirtual() || getBufferId() <= maxId;
    }
    
    /// Check if the capacity is within valid range
    bool hasValidCapacity() const {
      return getCapacityBytes() >= kMinCapacityBytes && 
             getCapacityBytes() <= kMaxCapacityBytes;
    }
    
    /// Check if a given size (in bytes) fits in this buffer
    bool canFit(int64_t sizeBytes) const {
      return sizeBytes <= getCapacityBytes();
    }
    
    /// Get the maximum number of elements that can fit
    int64_t getMaxElements(unsigned elementSizeBytes) const {
      if (elementSizeBytes == 0) return 0;
      return getCapacityBytes() / elementSizeBytes;
    }
    
    //===------------------------------------------------------------------===//
    // Factory Methods
    //===------------------------------------------------------------------===//
    
    /// Create a virtual (unallocated) buffer type
    static BufferType getVirtual(MLIRContext *ctx, Type elementType, 
                                  int64_t capacityBytes) {
      return get(ctx, elementType, kUnallocated, capacityBytes);
    }
    
    /// Create an allocated buffer type with specific ID
    static BufferType getAllocated(MLIRContext *ctx, Type elementType,
                                    unsigned bufferId, int64_t capacityBytes) {
      return get(ctx, elementType, bufferId, capacityBytes);
    }
    
    /// Create a new type with updated buffer ID (for allocation pass)
    BufferType withBufferId(unsigned newBufferId) const {
      return get(getContext(), getElementType(), newBufferId, getCapacityBytes());
    }
  }];
  
  let builders = [
    // Builder for virtual (unallocated) buffer
    TypeBuilder<(ins "Type":$elementType, "int64_t":$capacityBytes), [{
      return $_get($_ctxt, elementType, kUnallocated, capacityBytes);
    }]>,
    // Builder with default capacity for virtual buffer
    TypeBuilder<(ins "Type":$elementType), [{
      return $_get($_ctxt, elementType, kUnallocated,
                   mlir::allo::pynq::PYNQDialect::kDefaultBufferCapacityBytes);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Type Constraints for Operations
//===----------------------------------------------------------------------===//

// Constraint for any PYNQ buffer type (virtual or physical)
def AnyPYNQBuffer : Type<
  CPred<"llvm::isa<::mlir::allo::pynq::BufferType>($_self)">,
  "PYNQ buffer type">;

// Constraint for allocated (physical) buffer only
def AllocatedPYNQBuffer : Type<
  And<[
    CPred<"llvm::isa<::mlir::allo::pynq::BufferType>($_self)">,
    CPred<"llvm::cast<::mlir::allo::pynq::BufferType>($_self).isAllocated()">
  ]>,
  "allocated PYNQ buffer type">;

// Constraint for virtual (unallocated) buffer only
def VirtualPYNQBuffer : Type<
  And<[
    CPred<"llvm::isa<::mlir::allo::pynq::BufferType>($_self)">,
    CPred<"llvm::cast<::mlir::allo::pynq::BufferType>($_self).isVirtual()">
  ]>,
  "virtual PYNQ buffer type">;

#endif // PYNQ_TYPES
