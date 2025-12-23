/*
 * PYNQ Hardware Instruction Dialect - Operations
 * 
 * This file defines low-level hardware instruction operations that directly
 * correspond to the PYNQ accelerator's instruction encoding format.
 * These ops model the 32-bit instruction format from instr.h and provide
 * a thin MLIR abstraction over hardware capabilities.
 * 
 * Buffer Allocation Flow:
 * 1. Lowering: Generate pynq.buffer_alloc with virtual buffers (id=?)
 * 2. HoistBufferAllocPass: Move all allocs to function entry block
 * 3. BufferAllocationPass: Assign IDs, merge compatible buffers
 * 
 * Reference: externals/hyper_pynq_api/PYNQ_api/include/instr.h
 *   - gen_matmul_instr: 32-bit packed instruction for matrix multiplication
 *   - gen_data_instr: DMA data transfer instruction
 *   - gen_vector_instr: 32-bit packed instruction for vector operations
 * 
 * Instruction Field Widths (from instr.h):
 *   - BUFFER_ID_WIDTH: 3 bits (0-7)
 *   - TILE_ID_WIDTH: 3 bits (1-8 tiles)
 *   - PKG_ID_WIDTH: 12 bits (total package count)
 *   - REDUCE_K_ID_WIDTH: 8 bits (reduce axis element count)
 *   - HEAD_COUNT_ID_WIDTH: 2 bits (multi-head attention head count, 1-4)
 *   - HEAD_TILE_AXIS_ID_WIDTH: 1 bit (head tile axis selector)
 *   - ENABLE_BIAS_ID_WIDTH: 1 bit (bias enable flag)
 *   - ENABLE_TRANSPOSE_ID_WIDTH: 1 bit (transpose flag)
 *   - MATRIX_RESERVED_BITS_0: 4 bits (reserved)
 *   - OP_ID_WIDTH: 2 bits (vector operation type)
 *
 * Note: For Reduce K, Head Count and Tile Count fields, the hardware encoding uses (value - 1)
 * But you don't need to concern with this because it will be handled by the Runtime.
 * 
 * Note: Dialect and Type definitions are in PYNQDialect.td and PYNQTypes.td
 *       Attribute definitions are in PYNQAttrs.td
 */

include "allo/Dialect/PYNQDialect.td"
include "allo/Dialect/PYNQTypes.td"
include "allo/Dialect/PYNQAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// PYNQ Buffer Allocation Op
//===----------------------------------------------------------------------===//

def PYNQ_BufferAllocOp : PYNQ_Op<"buffer_alloc", []> {
  let summary = "Declare/allocate an on-chip buffer";
  let description = [{
    Declares an on-chip buffer for use by PYNQ operations. This op serves
    two purposes depending on the compilation stage:
    
    **Stage 1 (Virtual Declaration):**
    During lowering, buffer_alloc creates virtual buffers with unassigned IDs.
    These can be generated inline near their first use for convenience.
    ```mlir
    %buf = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
    ```
    
    **Stage 2 (After HoistBufferAllocPass):**
    All buffer_alloc ops are hoisted to the function entry block, making
    buffer lifetime analysis easier.
    ```mlir
    func.func @kernel(...) {
      %buf0 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      %buf1 = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      %buf2 = pynq.buffer_alloc : !pynq.buffer<i32, ?, 16384>
      // ... operations using buffers ...
    }
    ```
    
    **Stage 3 (After BufferAllocationPass):**
    Virtual buffers are assigned concrete IDs. Buffers with non-overlapping
    lifetimes may be merged to reduce physical buffer usage.
    ```mlir
    func.func @kernel(...) {
      %buf0 = pynq.buffer_alloc : !pynq.buffer<i8, 0, 4096>
      %buf1 = pynq.buffer_alloc : !pynq.buffer<i8, 1, 4096>
      %buf2 = pynq.buffer_alloc : !pynq.buffer<i32, 2, 16384>
      // ...
    }
    ```
    
    **Optional Attributes:**
    - role: Hint about buffer usage ("input", "weight", "output", "bias")
           Helps allocation pass make better decisions
    
    Examples:
      // Virtual buffer (during lowering)
      %buf = pynq.buffer_alloc : !pynq.buffer<i8, ?, 4096>
      
      // With role hint
      %input = pynq.buffer_alloc {role = "input"} : !pynq.buffer<i8, ?, 4096>
      
      // Allocated buffer (after allocation pass)
      %buf = pynq.buffer_alloc : !pynq.buffer<i8, 0, 4096>
  }];
  
  let arguments = (ins
    OptionalAttr<StrAttr>:$role    // Optional hint: "input", "weight", "output", "bias"
  );
  
  let results = (outs
    AnyPYNQBuffer:$buffer          // Returns buffer (virtual or physical)
  );
  
  let assemblyFormat = [{
    attr-dict `:` type($buffer)
  }];
  
  let builders = [
    // Builder for virtual buffer with capacity
    OpBuilder<(ins "Type":$elementType, "int64_t":$capacityBytes), [{
      auto bufferType = BufferType::getVirtual($_builder.getContext(), 
                                                elementType, capacityBytes);
      build($_builder, $_state, bufferType, /*role=*/nullptr);
    }]>,
    // Builder for virtual buffer with capacity and role
    OpBuilder<(ins "Type":$elementType, "int64_t":$capacityBytes, 
               "StringRef":$role), [{
      auto bufferType = BufferType::getVirtual($_builder.getContext(),
                                                elementType, capacityBytes);
      build($_builder, $_state, bufferType, $_builder.getStringAttr(role));
    }]>,
    // Builder for allocated buffer with specific ID
    OpBuilder<(ins "Type":$elementType, "unsigned":$bufferId, 
               "int64_t":$capacityBytes), [{
      auto bufferType = BufferType::getAllocated($_builder.getContext(),
                                                  elementType, bufferId, 
                                                  capacityBytes);
      build($_builder, $_state, bufferType, /*role=*/nullptr);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// Check if this is a virtual (unallocated) buffer
    bool isVirtual() {
      return getBuffer().getType().cast<BufferType>().isVirtual();
    }
    
    /// Check if this buffer has been allocated
    bool isAllocated() {
      return getBuffer().getType().cast<BufferType>().isAllocated();
    }
    
    /// Get the buffer ID (only valid if allocated)
    unsigned getBufferId() {
      return getBuffer().getType().cast<BufferType>().getBufferId();
    }
    
    /// Get the capacity in bytes
    int64_t getCapacityBytes() {
      return getBuffer().getType().cast<BufferType>().getCapacityBytes();
    }
    
    /// Get the element type
    Type getElementType() {
      return getBuffer().getType().cast<BufferType>().getElementType();
    }
    
    /// Check if this buffer has a role hint
    bool hasRole() {
      return getRole().has_value();
    }
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PYNQ MatMul Instruction Op
//===----------------------------------------------------------------------===//

def PYNQ_MatMulInstrOp : PYNQ_Op<"matmul_instr", []> {
  let summary = "Hardware matrix multiplication instruction";
  let description = [{
    Encodes a single PYNQ matmul instruction that executes on the accelerator.
    Maps directly to gen_matmul_instr() in instr.h.
    
    Instruction format (32 bits) - gen_matmul_instr():
      bits[0:2]   - RD: output_buffer_id (3 bits)
      bits[3:5]   - RS2: input_buffer_id (3 bits)  
      bits[6:8]   - RS1: weight_buffer_id (3 bits)
      bits[9:11]  - input_tile_count (3 bits, value 1-8)
      bits[12:14] - weight_tile_count (3 bits, value 1-8)
      bits[15:22] - reduce_k (8 bits, reduction dimension size, 1-256)
      bits[23:24] - head_count (2 bits, number of attention heads, 1-4)
      bits[25]    - head_tile_axis (1 bit, tiling axis for multi-head attention)
      bits[26:29] - reserved (4 bits)
      bit[30]     - enable_bias (1 bit, add bias vector: 0=no, 1=yes)
      bit[31]     - enable_transpose (1 bit, transpose output matrix: 0=no, 1=yes)
    
    Note:
      - 1 tile = 32 bytes = 32 x i8
      - Buffer IDs use RISC-V style naming: RS1 (input), RS2 (weight), RD (output)
      - buffer_msg array is packed in reverse order: [2]=RD, [1]=RS2, [0]=RS1
      - All parameters are passed as u32, matching instr.h function signature
    
    Hardware constraints:
    - Buffer IDs must be in range [0, 7]
    - Tile counts must be in range [1, 8]
    - reduce_k must be in range [1, 256]
    - head_count must be in range [1, 4]
    
    Example:
      pynq.matmul_instr 
        input_buf = %c0, weight_buf = %c1, output_buf = %c2,
        input_tiles = %c4, weight_tiles = %c4, reduce_k = %c32,
        head_count = %c0, head_tile_axis = %c0,
        enable_bias = %c0, enable_transpose = %c0
  }];
  
  let arguments = (ins
    I32:$input_buffer_id,      // RS1 (buffer_msg[0]): input buffer ID (0-7)
    I32:$weight_buffer_id,     // RS2 (buffer_msg[1]): weight buffer ID (0-7)
    I32:$output_buffer_id,     // RD (buffer_msg[2]): output buffer ID (0-7)
    I32:$input_tile_count,     // Input tile count (1-8)
    I32:$weight_tile_count,    // Weight tile count (1-8)
    I32:$reduce_k,             // Reduction dimension size (1-256)
    I32:$head_count,           // Number of attention heads (1-4)
    I32:$head_tile_axis,       // Tiling axis for multi-head attention (0 or 1)
    I32:$enable_bias,          // Add bias vector (0=no, 1=yes)
    I32:$enable_transpose      // Transpose weight matrix (0=no, 1=yes)
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `input_buf` `=` $input_buffer_id `,`
    `weight_buf` `=` $weight_buffer_id `,`
    `output_buf` `=` $output_buffer_id `,`
    `input_tiles` `=` $input_tile_count `,`
    `weight_tiles` `=` $weight_tile_count `,`
    `reduce_k` `=` $reduce_k `,`
    `head_count` `=` $head_count `,`
    `head_tile_axis` `=` $head_tile_axis `,`
    `enable_bias` `=` $enable_bias `,`
    `enable_transpose` `=` $enable_transpose
    attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PYNQ Data Transfer Op
//===----------------------------------------------------------------------===//

def PYNQ_DataTransferOp : PYNQ_Op<"data_transfer", []> {
  let summary = "DMA data transfer between host and device";
  let description = [{
    Transfers matrix data between host memory and on-chip buffer via DMA.
    Maps to gen_data_instr() in instr.h.
    
    Instruction format (32 bits) - gen_data_instr():
      bits[0:11]  - total_pkg_num (12 bits, total number of packages, 0-4095)
      bits[12:14] - tile_count (3 bits, value 1-8)
      bits[15:17] - buffer_id (3 bits, on-chip buffer ID 0-7)
      bits[18:31] - reserved (14 bits)

    Note:
      - 1 package = 256 bits = 32 bytes (one DMA transfer unit)
      - 1 tile = 32 bytes = 32 x int8
      - Direction is not part of the instruction encoding, but determines
        which hardware register to write to trigger the DMA
      - All parameters are passed as u32, matching instr.h function signature
    
    Direction:
      0 = host -> device (load to on-chip buffer)
      1 = device -> host (store from on-chip buffer)
    
    Hardware constraints:
    - Buffer ID must be in range [0, 7]
    - Tile count in range [1, 8]
    - total_pkg_num max is 4095 (12-bit limit)
    
    The memref operand contains the host-side data buffer. For load operations,
    data is read from memref and transferred to device. For store operations,
    data is transferred from device and written to memref.
    
    Example:
      // Load input matrix to buffer 0
      pynq.data_transfer %input_memref, buffer_id = %c0, tiles = %c4, 
                        total_pkg_num = %c128, direction = 0
      
      // Store result from buffer 2 back to host
      pynq.data_transfer %output_memref, buffer_id = %c2, tiles = %c4,
                        total_pkg_num = %c128, direction = 1
  }];
  
  let arguments = (ins
    AnyMemRef:$data,          // Host memory buffer
    I32:$buffer_id,           // On-chip buffer ID (0-7)
    I32:$tile_count,          // Number of tiles (1-8)
    I32:$total_pkg_num,       // Total number of packages to transfer
    I32:$direction            // 0 = host->device, 1 = device->host
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $data `,`
    `buffer_id` `=` $buffer_id `,`
    `tiles` `=` $tile_count `,`
    `total_pkg_num` `=` $total_pkg_num `,`
    `direction` `=` $direction
    attr-dict `:` type($data)
  }];
  
  let hasVerifier = 1;
}

// TODO: Add support for DDR to FIFO of Matrix/Vector

//===----------------------------------------------------------------------===//
// PYNQ Vector Instruction Base
//===----------------------------------------------------------------------===//
// 
// Vector instructions share a common encoding format - gen_vector_instr():
//   bits[0:2]   - buffer_id (3 bits, primary buffer ID)
//   bits[3:5]   - tile_count (3 bits, value 1-8)
//   bits[6:13]  - reduce_k (8 bits, 1-256)
//   bits[14:15] - op (2 bits, operation type)
//   bits[16:18] - extra_buffer_id (3 bits, secondary buffer for some ops)
//   bits[19:31] - reserved (13 bits)
//
// Operation types (op field):
//   0 = GELU
//   1 = QAdd (quantized element-wise addition)
//   2 = Softmax
//   3 = LayerNorm
//
// All parameters are passed as u32, matching instr.h function signature.
// We define separate ops for each vector operation type to provide
// a cleaner interface and hide the op encoding detail.
//===----------------------------------------------------------------------===//

// Base class for all vector operations (with extra_buffer_id)
class PYNQ_VectorOpBase<string mnemonic, string opSummary, int opCode> 
    : PYNQ_Op<mnemonic, []> {
  let summary = opSummary;
  
  let description = [{
    Vector operation mapped to gen_vector_instr() in instr.h.
    
    Instruction format (32 bits):
      bits[0:2]   - buffer_id (3 bits, primary buffer ID)
      bits[3:5]   - tile_count (3 bits, value 1-8)
      bits[6:13]  - reduce_k (8 bits, 1-256)
      bits[14:15] - op (2 bits, operation type)
      bits[16:18] - extra_buffer_id (3 bits, secondary buffer ID)
      bits[19:31] - reserved (13 bits)
  }];
  
  // Common arguments for vector ops (all u32 to match instr.h)
  let arguments = (ins
    I32:$buffer_id,           // Primary buffer ID (0-7)
    I32:$tile_count,          // Tile count (1-8)
    I32:$reduce_k,            // Reduce dimension size (1-256)
    I32:$extra_buffer_id      // Secondary buffer ID (0-7, for ops that need it)
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `buffer_id` `=` $buffer_id `,`
    `tiles` `=` $tile_count `,`
    `reduce_k` `=` $reduce_k `,`
    `extra_buffer_id` `=` $extra_buffer_id
    attr-dict
  }];
  
  int OpCode = opCode;
  
  let extraClassDeclaration = [{
    /// Get the operation code for this vector op
    static constexpr unsigned getOpCode() { return }] # opCode # [{; }
  }];
  
  let hasVerifier = 1;
}

// Base class for simple vector operations (without extra_buffer_id in IR)
// Inherits from PYNQ_VectorOpBase and overrides arguments/format
class PYNQ_SimpleVectorOpBase<string mnemonic, string opSummary, int opCode> 
    : PYNQ_VectorOpBase<mnemonic, opSummary, opCode> {
  
  let description = [{
    Simple vector operation mapped to gen_vector_instr() in instr.h.
    extra_buffer_id is not used and will be set to 0 during codegen.
    
    Instruction format (32 bits):
      bits[0:2]   - buffer_id (3 bits, primary buffer ID)
      bits[3:5]   - tile_count (3 bits, 1-8)
      bits[6:13]  - reduce_k (8 bits, 1-256)
      bits[14:15] - op (2 bits, operation type)
      bits[16:18] - extra_buffer_id = 0 (unused, 3 bits)
      bits[19:31] - reserved (13 bits)
  }];
  
  // Override: Arguments without extra_buffer_id
  let arguments = (ins
    I32:$buffer_id,           // Primary buffer ID (0-7)
    I32:$tile_count,          // Tile count (1-8)
    I32:$reduce_k             // Reduce dimension size (1-256)
  );
  
  // Override: Simplified assembly format
  let assemblyFormat = [{
    `buffer_id` `=` $buffer_id `,`
    `tiles` `=` $tile_count `,`
    `reduce_k` `=` $reduce_k
    attr-dict
  }];
  
  let extraClassDeclaration = [{
    /// Get the operation code for this vector op
    static constexpr unsigned getOpCode() { return }] # opCode # [{; }
    
    /// Get extra_buffer_id (always 0 for simple ops)
    static constexpr unsigned getExtraBufferId() { return 0; }
  }];
}

//===----------------------------------------------------------------------===//
// PYNQ GELU Op (op = 0)
//===----------------------------------------------------------------------===//

def PYNQ_GELUOp : PYNQ_SimpleVectorOpBase<"gelu", 
    "Hardware GELU activation operation", /*opCode=*/0> {
  let description = [{
    Performs GELU (Gaussian Error Linear Unit) activation on data in buffer.
    Maps to gen_vector_instr() with op = 0.
    
    Instruction format (32 bits) - gen_vector_instr():
      bits[0:2]   - buffer_id (3 bits, input/output buffer ID)
      bits[3:5]   - tile_count (3 bits, 1-8)
      bits[6:13]  - reduce_k (8 bits, 1-256)
      bits[14:15] - op = 0 (2 bits, GELU operation)
      bits[16:18] - extra_buffer_id (3 bits, unused, set to 0)
      bits[19:31] - reserved (13 bits)
    
    GELU is applied element-wise: GELU(x) = x * Φ(x)
    where Φ is the cumulative distribution function of standard normal.
    
    Example:
      pynq.gelu buffer_id = %c0, tiles = %c4, reduce_k = %c128
  }];
}

//===----------------------------------------------------------------------===//
// PYNQ QAdd Op (op = 1) - Quantized element-wise addition
//===----------------------------------------------------------------------===//

def PYNQ_QAddOp : PYNQ_VectorOpBase<"qadd", 
    "Hardware quantized element-wise addition operation", /*opCode=*/1> {
  let description = [{
    Performs quantized element-wise addition on data in on-chip buffers.
    Maps to gen_vector_instr() with op = 1.
    
    Instruction format (32 bits) - gen_vector_instr():
      bits[0:2]   - buffer_id (3 bits, first input and output buffer ID)
      bits[3:5]   - tile_count (3 bits, value 1-8)
      bits[6:13]  - reduce_k (8 bits, number of elements, 1-256)
      bits[14:15] - op = 1 (2 bits, QAdd operation)
      bits[16:18] - extra_buffer_id (3 bits, second input buffer ID)
      bits[19:31] - reserved (13 bits)
    
    The operation performs: buffer[buffer_id] += buffer[extra_buffer_id]
    Result is stored in buffer_id (in-place addition).
    
    Example:
      // Add buffer 1 to buffer 0: buf0 = buf0 + buf1
      pynq.qadd buffer_id = %c0, tiles = %c4, reduce_k = %c128,
                extra_buffer_id = %c1
  }];
}

//===----------------------------------------------------------------------===//
// PYNQ Softmax Op (op = 2)
//===----------------------------------------------------------------------===//

def PYNQ_SoftmaxOp : PYNQ_SimpleVectorOpBase<"softmax", 
    "Hardware softmax operation", /*opCode=*/2> {
  let description = [{
    Performs softmax operation on data in on-chip buffer.
    Maps to gen_vector_instr() with op = 2.
    
    Instruction format (32 bits) - gen_vector_instr():
      bits[0:2]   - buffer_id (3 bits, input/output buffer ID)
      bits[3:5]   - tile_count (3 bits, value 1-8)
      bits[6:13]  - reduce_k (8 bits, softmax dimension size, 1-256)
      bits[14:15] - op = 2 (2 bits, softmax operation)
      bits[16:18] - extra_buffer_id (3 bits, unused, set to 0)
      bits[19:31] - reserved (13 bits)
    
    The operation is performed in-place on the specified buffer.
    reduce_k specifies the dimension size for softmax normalization.
    
    Example:
      pynq.softmax buffer_id = %c0, tiles = %c4, reduce_k = %c32
  }];
}

//===----------------------------------------------------------------------===//
// PYNQ LayerNorm Op (op = 3)
//===----------------------------------------------------------------------===//

def PYNQ_LayerNormOp : PYNQ_SimpleVectorOpBase<"layernorm", 
    "Hardware layer normalization operation", /*opCode=*/3> {
  let description = [{
    Performs layer normalization on data in on-chip buffer.
    Maps to gen_vector_instr() with op = 3.
    
    Instruction format (32 bits) - gen_vector_instr():
      bits[0:2]   - buffer_id (3 bits, input/output buffer ID)
      bits[3:5]   - tile_count (3 bits, value 1-8)
      bits[6:13]  - reduce_k (8 bits, normalization dimension size, 1-256)
      bits[14:15] - op = 3 (2 bits, layernorm operation)
      bits[16:18] - extra_buffer_id (3 bits, unused, set to 0)
      bits[19:31] - reserved (13 bits)
    
    The operation normalizes the data in-place on the specified buffer.
    
    Example:
      pynq.layernorm buffer_id = %c0, tiles = %c4, reduce_k = %c64
  }];
}

//===----------------------------------------------------------------------===//
// PYNQ Generic Vector Op (for custom/future ops)
//===----------------------------------------------------------------------===//

def PYNQ_VectorOp : PYNQ_Op<"vector_op", []> {
  let summary = "Generic hardware vector operation";
  let description = [{
    Generic vector operation for custom or future vector instructions.
    Allows specifying the op code directly.
    Maps to gen_vector_instr() with configurable op field.
    
    Instruction format (32 bits) - gen_vector_instr():
      bits[0:2]   - buffer_id (3 bits, primary buffer ID)
      bits[3:5]   - tile_count (3 bits, value 1-8)
      bits[6:13]  - reduce_k (8 bits, 1-256)
      bits[14:15] - op (2 bits, operation type 0-3)
      bits[16:18] - extra_buffer_id (3 bits, secondary buffer ID)
      bits[19:31] - reserved (13 bits)
    
    Use this op when:
    - You need to specify a custom op code
    - New vector operations are added that don't have dedicated ops yet
    
    For standard operations, prefer using the dedicated ops:
    - pynq.gelu (op = 0)
    - pynq.qadd (op = 1)
    - pynq.softmax (op = 2)
    - pynq.layernorm (op = 3)
    
    Example:
      pynq.vector_op buffer_id = %c0, tiles = %c4, reduce_k = %c32,
                     op = %c1, extra_buffer_id = %c1
  }];
  
  let arguments = (ins
    I32:$buffer_id,           // Primary buffer ID (0-7)
    I32:$tile_count,          // Tile count (1-8)
    I32:$reduce_k,            // Operation parameter (1-256)
    I32:$op,                  // Operation type (0-3)
    I32:$extra_buffer_id      // Secondary buffer ID (0-7)
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `buffer_id` `=` $buffer_id `,`
    `tiles` `=` $tile_count `,`
    `reduce_k` `=` $reduce_k `,`
    `op` `=` $op `,`
    `extra_buffer_id` `=` $extra_buffer_id
    attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PYNQ Synchronization Op
//===----------------------------------------------------------------------===//

def PYNQ_SyncOp : PYNQ_Op<"sync", []> {
  let summary = "Synchronization barrier for accelerator completion";
  let description = [{
    Inserts a synchronization point that waits for all pending hardware
    operations to complete before proceeding. Maps to polling the
    accelerator status register or using interrupt-based synchronization.
    
    This is necessary before:
    1. Reading back results from device to host
    2. Issuing dependent operations on the same buffers
    3. Program termination

    Sync will be transformed into a a wait()-like call in the generated C++ code, such as while(status_register != 0) {...}
    
    Example:
      pynq.matmul_instr ...
      pynq.sync  // Wait for matmul to complete
      pynq.data_transfer ..., direction = 1  // Now safe to read result
  }];
  
  let arguments = (ins);
  let results = (outs);
  
  let assemblyFormat = [{ attr-dict }];
}

//===----------------------------------------------------------------------===//
// PYNQ CPU-side Quantize Op
//===----------------------------------------------------------------------===//

def PYNQ_QuantOp : PYNQ_Op<"quant", []> {
  let summary = "CPU-side (PS) quantization operation";
  let description = [{
    Performs floating-point to integer quantization on the ARM CPU (PS).
    This op runs on the CPU, not on the FPGA accelerator (PL).
    
    **Operation:**
    For symmetric mode: q = round(x / scale)
    For asymmetric mode: q = round(x / scale) + zero
    
    **Purpose:**
    Used at quantized/non-quantized boundaries where float tensors
    (e.g., cls_token, position_embeddings) need to be converted to
    integer format before being processed by quantized operations.
    
    **Quant Mode Encoding (2-bit):**
    - 0: symmetric, per-tensor
    - 1: asymmetric, per-tensor  
    - 2: symmetric, per-token
    - 3: asymmetric, per-token
    
    **Code Generation:**
    Will be lowered to CPU loop code that performs element-wise quantization.
    
    Example:
      pynq.quant output = %int_buf, input = %float_buf,
                 scale = %scale, quant_mode = 0
  }];
  
  let arguments = (ins
    AnyMemRef:$output,           // Output integer memref (pre-allocated)
    AnyMemRef:$input,            // Input float memref
    I32:$scale,                  // Packed scale (sign, coe, rshift in 32-bit)
    Optional<I32>:$zero,         // Zero point (for asymmetric mode)
    I8Attr:$quant_mode           // Quantization mode (0-3)
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `output` `=` $output `,` `input` `=` $input `,`
    `scale` `=` $scale
    (`zero` `=` $zero^)?
    `quant_mode` `=` $quant_mode attr-dict `:` 
    type($output) `,` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// PYNQ CPU-side Dequantize Op
//===----------------------------------------------------------------------===//

def PYNQ_DequantOp : PYNQ_Op<"dequant", []> {
  let summary = "CPU-side (PS) dequantization operation";
  let description = [{
    Performs integer to floating-point dequantization on the ARM CPU (PS).
    This op runs on the CPU, not on the FPGA accelerator (PL).
    
    **Operation:**
    For symmetric mode: x = q * scale
    For asymmetric mode: x = (q - zero) * scale
    
    **Purpose:**
    Used at quantized/non-quantized boundaries where integer tensors
    need to be converted back to float format for operations that
    require floating-point precision (e.g., final output, comparison).
    
    **Quant Mode Encoding (2-bit):**
    - 0: symmetric, per-tensor
    - 1: asymmetric, per-tensor
    - 2: symmetric, per-token
    - 3: asymmetric, per-token
    
    **Code Generation:**
    Will be lowered to CPU loop code that performs element-wise dequantization.
    
    Example:
      pynq.dequant output = %float_buf, input = %int_buf,
                   scale = %scale, quant_mode = 0
  }];
  
  let arguments = (ins
    AnyMemRef:$output,           // Output float memref (pre-allocated)
    AnyMemRef:$input,            // Input integer memref
    I32:$scale,                  // Packed scale (sign, coe, rshift in 32-bit)
    Optional<I32>:$zero,         // Zero point (for asymmetric mode)
    I8Attr:$quant_mode           // Quantization mode (0-3)
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `output` `=` $output `,` `input` `=` $input `,`
    `scale` `=` $scale
    (`zero` `=` $zero^)?
    `quant_mode` `=` $quant_mode attr-dict `:` 
    type($output) `,` type($input)
  }];
}

// //===----------------------------------------------------------------------===//
// // PYNQ Interleave Op (Runtime data layout transformation)
// //===----------------------------------------------------------------------===//

// def PYNQ_InterleaveOp : PYNQ_Op<"interleave", [SameOperandsAndResultType]> {
//   let summary = "Runtime data layout transformation for hardware-efficient access";
//   let description = [{
//     Transforms data layout at runtime for efficient hardware access patterns.
//     This op is inserted during vivado → pynq lowering for DYNAMIC data
//     (e.g., activations) that cannot be pre-transformed at compile time.
    
//     **When to use this op vs static_interleave attribute:**
    
//     - `pynq.interleave` op: For runtime/dynamic data (activations, intermediate 
//       results). Generates actual transformation code.
      
//     - `pynq.static_interleave` attribute on memref.global: For compile-time 
//       known data (weights, biases). The export tool transforms data offline,
//       no runtime overhead.
    
//     **Interleave Patterns:**
//     - none: No transformation (passthrough)
//     - row_major_tiled: Rearrange to row-major tiles
//     - col_major_tiled: Rearrange to column-major tiles  
//     - weight_interleave: Special pattern for weight matrices
//     - custom: User-defined pattern (specify via custom_pattern attribute)
    
//     **Code Generation:**
//     This op will be lowered to nested loops that perform the rearrangement:
//     ```cpp
//     for (tile_row = 0; tile_row < M/tile_h; tile_row++)
//       for (tile_col = 0; tile_col < N/tile_w; tile_col++)
//         for (i = 0; i < tile_h; i++)
//           for (j = 0; j < tile_w; j++)
//             output[...] = input[...];  // Pattern-specific indexing
//     ```
    
//     **Example:**
//     ```mlir
//     // Transform activation for hardware access pattern
//     %interleaved = pynq.interleave %activation 
//         {pattern = #pynq<interleave_pattern weight_interleave>}
//         : memref<128x128xi8>
    
//     // Then transfer the interleaved data
//     pynq.data_transfer %interleaved, buffer_id = 0, ...
//     ```
//   }];
  
//   let arguments = (ins
//     AnyMemRef:$input,                                    // Input data (original layout)
//     PYNQ_InterleavePatternAttr:$pattern,                 // Interleave pattern
//     OptionalAttr<I64ArrayAttr>:$tile_sizes,              // Optional tile sizes [tile_h, tile_w]
//     OptionalAttr<StrAttr>:$custom_pattern                // Custom pattern specification
//   );
  
//   let results = (outs
//     AnyMemRef:$output                                    // Output data (interleaved layout)
//   );
  
//   let assemblyFormat = [{
//     $input attr-dict `:` type($input)
//   }];
  
//   let builders = [
//     // Simple builder with just pattern
//     OpBuilder<(ins "Value":$input, "InterleavePattern":$pattern), [{
//       build($_builder, $_state, input.getType(), input,
//             PYNQ_InterleavePatternAttr::get($_builder.getContext(), pattern),
//             nullptr, nullptr);
//     }]>
//   ];
  
//   let extraClassDeclaration = [{
//     /// Check if this is a no-op (pattern = none)
//     bool isNoop() {
//       return getPattern() == InterleavePattern::none;
//     }
    
//     /// Check if this uses a custom pattern
//     bool hasCustomPattern() {
//       return getPattern() == InterleavePattern::custom && 
//              getCustomPattern().has_value();
//     }
//   }];
// }

// //===----------------------------------------------------------------------===//
// // PYNQ Deinterleave Op (Reverse transformation for output)
// //===----------------------------------------------------------------------===//

// def PYNQ_DeinterleaveOp : PYNQ_Op<"deinterleave", [SameOperandsAndResultType]> {
//   let summary = "Reverse runtime data layout transformation";
//   let description = [{
//     Transforms data from hardware layout back to standard layout.
//     This is the inverse operation of pynq.interleave.
    
//     Used when:
//     1. Reading output data from device that was stored in interleaved format
//     2. The consumer expects standard (non-interleaved) layout
    
//     For static outputs where the consumer can handle interleaved layout,
//     this op can be elided.
    
//     Example:
//     ```mlir
//     // Read interleaved output from device
//     pynq.data_transfer %output_buffer, buffer_id = 2, ..., direction = 1
    
//     // Transform back to standard layout if needed
//     %result = pynq.deinterleave %output_buffer
//         {pattern = #pynq<interleave_pattern row_major_tiled>}
//         : memref<128x128xi32>
//     ```
//   }];
  
//   let arguments = (ins
//     AnyMemRef:$input,
//     PYNQ_InterleavePatternAttr:$pattern,
//     OptionalAttr<I64ArrayAttr>:$tile_sizes,
//     OptionalAttr<StrAttr>:$custom_pattern
//   );
  
//   let results = (outs
//     AnyMemRef:$output
//   );
  
//   let assemblyFormat = [{
//     $input attr-dict `:` type($input)
//   }];
// }