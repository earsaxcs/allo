/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * PYNQ Buffer Allocation Pass
 * 
 * This pass transforms virtual buffer allocations (bufferId = ?) into
 * physical buffer allocations (bufferId = 0-7) by performing:
 * 1. Liveness analysis of buffer usage
 * 2. Linear scan register allocation for buffer IDs
 * 3. Buffer merging for non-overlapping lifetimes
 * 4. (Optional) Spill insertion when all buffers are exhausted
 * 
 * Input: pynq.buffer_alloc ops with virtual buffers (!pynq.buffer<i8, ?, 4096>)
 * Output: pynq.buffer_alloc ops with physical buffers (!pynq.buffer<i8, 0, 4096>)
 * 
 * The pass should be run after PYNQHoistBufferAllocPass and before code generation.
 */

#include "allo/Transforms/Passes.h"
#include "allo/Dialect/PYNQDialect.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SetVector.h"

#include <queue>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

//===----------------------------------------------------------------------===//
// Hardware Configuration
//===----------------------------------------------------------------------===//

namespace {

/// Hardware configuration parameters for buffer allocation.
/// These can be overridden via pass options for different PYNQ targets.
struct PYNQHardwareConfig {
  /// Number of on-chip buffers available (default: 8 for 3-bit buffer ID)
  unsigned numBuffers = BufferConfig::kNumBuffers;
  
  /// Default buffer capacity in bytes
  int64_t defaultBufferCapacity = BufferConfig::kDefaultCapacityBytes;
  
  /// Maximum buffer capacity in bytes (hardware limit)
  int64_t maxBufferCapacity = BufferConfig::kMaxCapacityBytes;
  
  /// Tile size in elements (for computing buffer requirements)
  unsigned tileSize = TileConfig::kElementsPerTile;
  
  /// Whether to enable buffer spilling when all buffers are occupied
  bool enableSpilling = false;
  
  /// Validate the configuration
  bool isValid() const {
    return numBuffers > 0 && numBuffers <= 8 &&
           defaultBufferCapacity > 0 &&
           defaultBufferCapacity <= maxBufferCapacity;
  }
};

//===----------------------------------------------------------------------===//
// Liveness Analysis Data Structures
//===----------------------------------------------------------------------===//

/// Represents a live range for a virtual buffer allocation.
/// The range is defined by instruction indices in a linearized program order.
struct LiveRange {
  /// The buffer_alloc operation (with virtual buffer)
  BufferAllocOp allocOp;
  
  /// Start index (where the buffer is allocated)
  unsigned startIndex;
  
  /// End index (last use of the buffer value)
  unsigned endIndex;
  
  /// Required buffer size in bytes
  int64_t requiredSizeBytes;
  
  /// Element type of the buffer
  Type elementType;
  
  /// Buffer role hint (input/weight/output/bias), if available
  std::optional<StringRef> role;
  
  /// Assigned buffer ID (set by allocation algorithm)
  std::optional<unsigned> assignedBufferId;
  
  /// Whether this allocation requires spilling
  bool needsSpill = false;
  
  /// Check if this range overlaps with another
  bool overlaps(const LiveRange &other) const {
    return !(endIndex < other.startIndex || other.endIndex < startIndex);
  }
  
  /// Check if this range is active at a given index
  bool isActiveAt(unsigned index) const {
    return startIndex <= index && index <= endIndex;
  }
};

/// Represents a buffer assignment decision
struct BufferAssignment {
  unsigned bufferId;
  Type elementType;
  int64_t capacityBytes;
  
  /// The live ranges currently assigned to this buffer
  /// (non-overlapping by construction)
  SmallVector<LiveRange *, 4> assignedRanges;
  
  /// Check if a range can be assigned to this buffer
  bool canAssign(const LiveRange &range) const {
    for (auto *existing : assignedRanges) {
      if (existing->overlaps(range)) {
        return false;
      }
    }
    return range.requiredSizeBytes <= capacityBytes;
  }
};

//===----------------------------------------------------------------------===//
// Buffer Allocation Analysis
//===----------------------------------------------------------------------===//

/// Performs liveness analysis and buffer allocation for a function.
class BufferAllocationAnalysis {
public:
  BufferAllocationAnalysis(func::FuncOp funcOp, 
                           const PYNQHardwareConfig &config)
      : funcOp(funcOp), config(config) {}
  
  /// Run the complete analysis
  LogicalResult analyze();
  
  /// Get the allocation results
  const SmallVector<LiveRange> &getLiveRanges() const { return liveRanges; }
  
private:
  func::FuncOp funcOp;
  const PYNQHardwareConfig &config;
  
  /// All live ranges discovered during analysis
  SmallVector<LiveRange> liveRanges;
  
  /// Mapping from Operation* to linear instruction index
  DenseMap<Operation *, unsigned> opToIndex;
  
  /// Linearize the operations in program order
  void buildLinearOrder();
  
  /// Collect all buffer requests and compute their live ranges
  LogicalResult collectLiveRanges();
  
  /// Find the last use of a value
  unsigned findLastUse(Value value);
};

//===----------------------------------------------------------------------===//
// Buffer Allocator (Linear Scan Algorithm)
//===----------------------------------------------------------------------===//

/// Implements linear scan register allocation for buffer IDs.
/// 
/// Algorithm overview:
/// 1. Sort live ranges by start index
/// 2. Maintain a set of active ranges and free buffers
/// 3. For each range in order:
///    a. Expire old ranges that have ended
///    b. Try to assign a free buffer
///    c. If no buffer available, spill (if enabled) or fail
class LinearScanAllocator {
public:
  LinearScanAllocator(const PYNQHardwareConfig &config)
      : config(config), freeBuffers(config.numBuffers, true) {}
  
  /// Run allocation on the given live ranges
  /// Returns success if all ranges were allocated, failure otherwise
  LogicalResult allocate(SmallVector<LiveRange> &ranges);
  
  /// Get the buffer assignments
  const SmallVector<BufferAssignment> &getAssignments() const { 
    return assignments; 
  }
  
private:
  const PYNQHardwareConfig &config;
  
  /// Bit vector tracking which buffer IDs are free
  llvm::SmallBitVector freeBuffers;
  
  /// Currently active live ranges (sorted by end index for expiration)
  std::priority_queue<
      std::pair<unsigned, LiveRange *>,
      std::vector<std::pair<unsigned, LiveRange *>>,
      std::greater<std::pair<unsigned, LiveRange *>>> activeRanges;
  
  /// Buffer assignments
  SmallVector<BufferAssignment> assignments;
  
  /// Expire ranges that have ended before the given index
  void expireOldRanges(unsigned currentIndex);
  
  /// Try to allocate a buffer for the given range
  std::optional<unsigned> tryAllocate(LiveRange &range);
  
  /// Select a range to spill when no buffers are available
  LiveRange *selectSpillCandidate(const LiveRange &newRange);
  
  /// Free a buffer ID
  void freeBuffer(unsigned bufferId);
  
  /// Allocate a specific buffer ID
  void allocateBuffer(unsigned bufferId, LiveRange &range);
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class PYNQBufferAllocationPass 
    : public PassWrapper<PYNQBufferAllocationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQBufferAllocationPass)
  
  StringRef getArgument() const override { return "pynq-buffer-allocation"; }
  StringRef getDescription() const override {
    return "Allocate PYNQ on-chip buffers for buffer requests";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
  }
  
  void runOnOperation() override;
  
private:
  /// Hardware configuration (can be set via pass options)
  PYNQHardwareConfig hwConfig;
  
  /// Process a single function
  LogicalResult processFunction(func::FuncOp funcOp);
  
  /// Transform buffer_request ops to buffer_alloc ops based on analysis
  LogicalResult applyAllocations(func::FuncOp funcOp,
                                  SmallVector<LiveRange> &ranges);
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// BufferAllocationAnalysis Implementation
//===----------------------------------------------------------------------===//

void BufferAllocationAnalysis::buildLinearOrder() {
  /*
   * 设计思路:
   * 遍历函数中的所有操作，按照程序执行顺序分配索引。
   * 
   * 对于简单的顺序代码，直接按 block 内顺序即可。
   * 对于包含 scf.for/scf.if 等控制流的代码：
   * - 循环体内的操作需要特殊处理（整个循环视为一个活跃区间）
   * - 条件分支需要合并两个分支的活跃信息
   * 
   * 当前简化实现：只处理单基本块的情况
   * TODO: 支持嵌套区域（loops, conditionals）
   */
  
  unsigned index = 0;
  funcOp.walk([&](Operation *op) {
    opToIndex[op] = index++;
  });
}

LogicalResult BufferAllocationAnalysis::collectLiveRanges() {
  /*
   * 设计思路:
   * 1. 遍历所有 pynq.buffer_alloc 操作
   * 2. 过滤出虚拟 buffer (bufferId = kUnallocated)
   * 3. 对于每个虚拟 buffer_alloc：
   *    - startIndex = alloc 操作的位置
   *    - endIndex = buffer value 最后一次被使用的位置
   *    - requiredSizeBytes = buffer type 的 capacity
   *    - elementType = buffer type 的 element type
   *    - role = alloc 的 role 属性（可选）
   * 
   * 注意：如果 buffer value 没有被使用（dead code），endIndex = startIndex
   */
  
  funcOp.walk([&](BufferAllocOp allocOp) {
    // Only process virtual buffers
    if (!allocOp.isVirtual()) {
      return;
    }
    
    LiveRange range;
    range.allocOp = allocOp;
    range.startIndex = opToIndex[allocOp.getOperation()];
    range.endIndex = findLastUse(allocOp.getBuffer());
    range.requiredSizeBytes = allocOp.getCapacityBytes();
    range.elementType = allocOp.getElementType();
    
    if (allocOp.hasRole()) {
      range.role = allocOp.getRole();
    }
    
    liveRanges.push_back(range);
  });
  
  return success();
}

unsigned BufferAllocationAnalysis::findLastUse(Value value) {
  /*
   * 设计思路:
   * 遍历 value 的所有 uses，找到最大的指令索引。
   * 
   * 特殊情况：
   * - 如果 value 没有 use，返回定义点的索引
   * - 如果 value 在循环中使用，需要考虑循环的结束点
   *   （当前简化：不特殊处理循环）
   */
  
  unsigned lastUse = opToIndex[value.getDefiningOp()];
  
  for (Operation *user : value.getUsers()) {
    if (auto it = opToIndex.find(user); it != opToIndex.end()) {
      lastUse = std::max(lastUse, it->second);
    }
  }
  
  return lastUse;
}

LogicalResult BufferAllocationAnalysis::analyze() {
  buildLinearOrder();
  return collectLiveRanges();
}

//===----------------------------------------------------------------------===//
// LinearScanAllocator Implementation
//===----------------------------------------------------------------------===//

void LinearScanAllocator::expireOldRanges(unsigned currentIndex) {
  /*
   * 设计思路:
   * 从 activeRanges 优先队列中移除所有 endIndex < currentIndex 的 range，
   * 并释放它们占用的 buffer ID。
   * 
   * activeRanges 按 endIndex 升序排列，所以可以快速移除过期的 ranges。
   */
  
  while (!activeRanges.empty() && 
         activeRanges.top().first < currentIndex) {
    auto [endIdx, range] = activeRanges.top();
    activeRanges.pop();
    
    if (range->assignedBufferId) {
      freeBuffer(*range->assignedBufferId);
    }
  }
}

std::optional<unsigned> LinearScanAllocator::tryAllocate(LiveRange &range) {
  /*
   * 设计思路:
   * 1. 找到第一个空闲的 buffer ID
   * 2. 检查该 buffer 的容量是否满足需求
   * 3. 如果满足，分配并返回 buffer ID
   * 4. 如果没有空闲 buffer，返回 nullopt
   * 
   * 优化机会：
   * - 可以根据 role hint 优先选择特定的 buffer（如 input 总是用 0-2）
   * - 可以实现最小碎片化的选择策略
   */
  
  // 简单策略：找第一个空闲的 buffer
  for (unsigned i = 0; i < config.numBuffers; ++i) {
    if (freeBuffers[i]) {
      // 检查容量
      if (range.requiredSizeBytes <= config.defaultBufferCapacity) {
        allocateBuffer(i, range);
        return i;
      }
    }
  }
  
  return std::nullopt;
}

LiveRange *LinearScanAllocator::selectSpillCandidate(const LiveRange &newRange) {
  /*
   * 设计思路（当启用 spilling 时）:
   * 选择一个当前活跃的 range 进行 spill（换出到主存）。
   * 
   * 选择策略：
   * 1. 优先选择 endIndex 最远的（减少后续冲突）
   * 2. 或选择 size 最大的（释放更多空间）
   * 3. 或根据 role 选择（output 比 input 更容易 spill）
   * 
   * 当前简化：选择 endIndex 最远的
   * 
   * Spilling 实现：
   * - 在 spill 点插入 pynq.data_transfer (direction=1) 将数据写回
   * - 在重新需要时插入 pynq.data_transfer (direction=0) 重新加载
   * - 标记该 range 的 needsSpill = true
   */
  
  // TODO: 实现 spill 候选选择
  // 当前返回 nullptr 表示不支持 spilling
  return nullptr;
}

void LinearScanAllocator::freeBuffer(unsigned bufferId) {
  freeBuffers.set(bufferId);
}

void LinearScanAllocator::allocateBuffer(unsigned bufferId, LiveRange &range) {
  freeBuffers.reset(bufferId);
  range.assignedBufferId = bufferId;
  activeRanges.push({range.endIndex, &range});
}

LogicalResult LinearScanAllocator::allocate(SmallVector<LiveRange> &ranges) {
  /*
   * 设计思路（线性扫描算法）:
   * 
   * 1. 将所有 live ranges 按 startIndex 排序
   * 2. 依次处理每个 range：
   *    a. expireOldRanges: 释放已经结束的 ranges 占用的 buffer
   *    b. tryAllocate: 尝试分配一个空闲 buffer
   *    c. 如果分配失败且启用 spilling：选择一个 victim 进行 spill
   *    d. 如果分配失败且不支持 spilling：报错
   * 
   * 复杂度: O(n log n) 排序 + O(n log n) 处理（优先队列操作）
   */
  
  // Step 1: Sort by start index
  llvm::sort(ranges, [](const LiveRange &a, const LiveRange &b) {
    return a.startIndex < b.startIndex;
  });
  
  // Step 2: Process each range in order
  for (LiveRange &range : ranges) {
    // Expire old ranges
    expireOldRanges(range.startIndex);
    
    // Try to allocate
    auto bufferId = tryAllocate(range);
    
    if (!bufferId) {
      // No free buffer available
      if (config.enableSpilling) {
        // TODO: Implement spilling
        LiveRange *victim = selectSpillCandidate(range);
        if (victim) {
          victim->needsSpill = true;
          freeBuffer(*victim->assignedBufferId);
          bufferId = tryAllocate(range);
        }
      }
      
      if (!bufferId) {
        // Allocation failed
        range.allocOp.emitError() 
            << "failed to allocate buffer: all " << config.numBuffers 
            << " buffers are occupied and spilling is "
            << (config.enableSpilling ? "not possible" : "disabled");
        return failure();
      }
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void PYNQBufferAllocationPass::runOnOperation() {
  /*
   * 设计思路:
   * 1. 遍历 module 中的所有函数
   * 2. 对每个函数执行 buffer 分配
   * 3. 将 pynq.buffer_request 替换为 pynq.buffer_alloc
   */
  
  ModuleOp moduleOp = getOperation();
  
  // Validate hardware configuration
  if (!hwConfig.isValid()) {
    moduleOp.emitError() << "invalid PYNQ hardware configuration";
    return signalPassFailure();
  }
  
  // Process each function
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (failed(processFunction(funcOp))) {
      return signalPassFailure();
    }
  }
}

LogicalResult PYNQBufferAllocationPass::processFunction(func::FuncOp funcOp) {
  /*
   * 设计思路:
   * 1. 检查函数是否包含虚拟 buffer_alloc 操作
   * 2. 如果没有虚拟 buffer，直接返回（无需处理）
   * 3. 执行 liveness 分析
   * 4. 执行 buffer ID 分配
   * 5. 更新 buffer_alloc 操作的类型（virtual → physical）
   */
  
  // Quick check: does this function have any virtual buffers?
  bool hasVirtualBuffers = false;
  funcOp.walk([&](BufferAllocOp allocOp) {
    if (allocOp.isVirtual()) {
      hasVirtualBuffers = true;
    }
  });
  
  if (!hasVirtualBuffers) {
    return success();
  }
  
  // Run liveness analysis
  BufferAllocationAnalysis analysis(funcOp, hwConfig);
  if (failed(analysis.analyze())) {
    return funcOp.emitError() << "liveness analysis failed";
  }
  
  // Get live ranges (need a mutable copy for allocation)
  SmallVector<LiveRange> ranges = analysis.getLiveRanges();
  
  if (ranges.empty()) {
    return success();
  }
  
  // Run buffer allocation
  LinearScanAllocator allocator(hwConfig);
  if (failed(allocator.allocate(ranges))) {
    return failure(); // Error already emitted
  }
  
  // Apply allocations to IR
  return applyAllocations(funcOp, ranges);
}

LogicalResult PYNQBufferAllocationPass::applyAllocations(
    func::FuncOp funcOp,
    SmallVector<LiveRange> &ranges) {
  /*
   * 设计思路:
   * 对于每个已分配的 live range：
   * 1. 更新 buffer_alloc 操作的结果类型（从 virtual 变为 physical）
   * 2. 保持 buffer value 的所有使用不变（因为类型兼容）
   * 
   * 如果 range.needsSpill = true：
   * - 在适当位置插入 spill store (data_transfer direction=1)
   * - 在重新需要时插入 spill load (data_transfer direction=0)
   * 
   * IR 变换示例：
   * 
   * Before:
   *   %buf = pynq.buffer_alloc {role = "input"} : !pynq.buffer<i8, ?, 4096>
   *   pynq.data_transfer %memref, buffer_id = ..., ...
   * 
   * After:
   *   %buf = pynq.buffer_alloc {role = "input"} : !pynq.buffer<i8, 0, 4096>
   *   pynq.data_transfer %memref, buffer_id = ..., ...
   * 
   * Note: The buffer value %buf is still used by the same operations.
   * We only change the type from virtual to physical.
   */
  
  OpBuilder builder(funcOp.getContext());
  
  for (LiveRange &range : ranges) {
    if (!range.assignedBufferId) {
      return range.allocOp.emitError() << "buffer was not allocated";
    }
    
    BufferAllocOp allocOp = range.allocOp;
    unsigned bufferId = *range.assignedBufferId;
    
    // Create new physical buffer type
    auto physicalBufferType = BufferType::getAllocated(
        allocOp.getContext(),
        range.elementType,
        bufferId,
        range.requiredSizeBytes);
    
    // Create new buffer_alloc with physical buffer type
    builder.setInsertionPoint(allocOp);
    auto newAllocOp = builder.create<BufferAllocOp>(
        allocOp.getLoc(),
        range.elementType,
        bufferId,
        range.requiredSizeBytes);
    
    // Copy role attribute if present
    if (allocOp.hasRole()) {
      newAllocOp.setRoleAttr(allocOp.getRoleAttr());
    }
    
    // Replace all uses of the old virtual buffer with the new physical buffer
    allocOp.getBuffer().replaceAllUsesWith(newAllocOp.getBuffer());
    
    // Handle spilling if needed
    if (range.needsSpill) {
      // TODO: Insert spill operations
      // This requires:
      // 1. Finding the point where the buffer is evicted
      // 2. Inserting data_transfer to save to host memory
      // 3. Finding the point where it's reloaded
      // 4. Inserting data_transfer to load from host memory
      allocOp.emitWarning() << "spilling not yet implemented";
    }
    
    // Erase the original virtual buffer_alloc
    allocOp.erase();
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createPYNQBufferAllocationPass() {
  return std::make_unique<PYNQBufferAllocationPass>();
}

} // namespace allo
} // namespace mlir
