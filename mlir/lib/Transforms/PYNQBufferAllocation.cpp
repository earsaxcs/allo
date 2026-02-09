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
 * The pass should be run before code generation. If paired with
 * PYNQHoistBufferAllocPass, run this pass first to assign IDs and insert
 * spills, then hoist/dedupe allocations.
 */

#include "PassDetail.h"
#include "allo/Transforms/Passes.h"
#include "allo/Dialect/PYNQTypes.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Dialect/PYNQConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"

#include <optional>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::pynq;

//===----------------------------------------------------------------------===//
// Hardware Configuration
//===----------------------------------------------------------------------===//

namespace {

/// Buffer allocation strategy for LinearScanAllocator.
enum class AllocationStrategy {
  /// Save buffers: allocate first available buffer (original behavior).
  /// Prioritizes minimizing buffer count.
  SaveBuffers = 0,
  
  /// Balanced utilization: balance buffer usage and reduce adjacent conflicts.
  /// Tries to maximize buffer utilization while avoiding immediate reuse
  /// of buffers released by adjacent operations.
  BalancedUtilization = 1
};

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
  
  /// Buffer allocation strategy
  AllocationStrategy strategy = AllocationStrategy::SaveBuffers;
  
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

/// A live interval segment for a single virtual buffer value.
///
/// We allow splitting a virtual buffer's lifetime into multiple segments when
/// spilling is required. Each segment will be rewritten to use its own physical
/// `pynq.buffer_alloc` SSA value (possibly with different buffer IDs).
struct LiveSegment {
  BufferAllocOp virtualAlloc;
  Value virtualBuffer;

  unsigned startIndex;
  unsigned endIndex;

  int64_t requiredSizeBytes;
  Type elementType;
  std::optional<StringRef> role;

  std::optional<unsigned> assignedBufferId;

  // Spill bookkeeping.
  bool needsStoreToSlot = false;
  unsigned storeBeforeIndex = 0;
  bool needsReloadFromSlot = false;

  // Filled during IR rewrite.
  BufferAllocOp physicalAlloc;

  bool overlaps(const LiveSegment &other) const {
    return !(endIndex < other.startIndex || other.endIndex < startIndex);
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
  const SmallVector<LiveSegment> &getSegments() const { return segments; }

  Block *getSingleBlock() const { return singleBlock; }

  Operation *getOpAt(unsigned index) const {
    if (index >= indexToOp.size())
      return nullptr;
    return indexToOp[index];
  }

  std::optional<unsigned> getIndexOf(Operation *op) const {
    auto it = opToIndex.find(op);
    if (it == opToIndex.end())
      return std::nullopt;
    return it->second;
  }
  
private:
  func::FuncOp funcOp;
  const PYNQHardwareConfig &config;
  
  /// All live segments discovered during analysis
  SmallVector<LiveSegment> segments;
  
  /// Mapping from Operation* to linear instruction index
  DenseMap<Operation *, unsigned> opToIndex;

  /// Reverse map for stable insertion points
  SmallVector<Operation *> indexToOp;

  /// We only support a single basic block in the first version.
  Block *singleBlock = nullptr;
  
  /// Linearize the operations in program order
  void buildLinearOrder();
  
  /// Collect all virtual buffers and compute their [firstUse,lastUse]
  LogicalResult collectSegments();

  /// Find first/last use of a value inside `singleBlock`.
  std::optional<std::pair<unsigned, unsigned>>
  findFirstLastUseInBlock(Value value);
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
  LinearScanAllocator(const PYNQHardwareConfig &config,
                      const BufferAllocationAnalysis &analysis)
      : config(config), analysis(analysis), 
        freeBuffers(config.numBuffers, true),
        bufferUsageCount(config.numBuffers, 0),
        lastReleaseIndex(config.numBuffers, 0) {}
  
  /// Run allocation on the given live ranges
  /// Returns success if all ranges were allocated, failure otherwise
  LogicalResult allocate(SmallVector<LiveSegment> &segments,
                         const BufferAllocationAnalysis &analysis);
  
private:
  const PYNQHardwareConfig &config;
  const BufferAllocationAnalysis &analysis;
  
  /// Bit vector tracking which buffer IDs are free
  llvm::SmallBitVector freeBuffers;

  /// Currently active segments.
  SmallVector<LiveSegment *, 16> active;
  
  /// Buffer usage count (for balanced utilization strategy)
  SmallVector<unsigned, 8> bufferUsageCount;
  
  /// Last release index for each buffer (for adjacent conflict detection)
  SmallVector<unsigned, 8> lastReleaseIndex;

  void expireOld(unsigned currentIndex);

  std::optional<unsigned> tryAllocate(LiveSegment &segment);
  
  std::optional<unsigned> tryAllocateSaveBuffers(LiveSegment &segment);
  
  std::optional<unsigned> tryAllocateBalancedUtilization(LiveSegment &segment);

  LiveSegment *selectSpillVictim(Operation *currentOp,
                                ArrayRef<Value> currentOpBuffers) const;
  
  /// Free a buffer ID
  void freeBuffer(unsigned bufferId);
  
  /// Allocate a specific buffer ID
  void allocateBuffer(unsigned bufferId, LiveSegment &segment);
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class PYNQBufferAllocationPass 
    : public PassWrapper<PYNQBufferAllocationPass, OperationPass<ModuleOp>> {
public:
  PYNQBufferAllocationPass() = default;
  PYNQBufferAllocationPass(const PYNQBufferAllocationPass &pass)
      : PassWrapper(pass) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQBufferAllocationPass)
  
  StringRef getArgument() const override { return "pynq-buffer-allocation"; }
  StringRef getDescription() const override {
    return "Allocate PYNQ on-chip buffers for buffer requests";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
  }

    Option<unsigned> numBuffers{
      *this, "num-buffers",
      llvm::cl::desc("Number of on-chip buffers available"),
      llvm::cl::init(BufferConfig::kNumBuffers)};

    Option<int64_t> bufferCapacity{
      *this, "buffer-capacity",
      llvm::cl::desc("Default buffer capacity in bytes"),
      llvm::cl::init(BufferConfig::kDefaultCapacityBytes)};

    Option<bool> enableSpilling{
      *this, "enable-spilling",
      llvm::cl::desc("Enable buffer spilling when buffers are exhausted"),
      llvm::cl::init(false)};
    
    Option<AllocationStrategy> allocationStrategy{
      *this, "allocation-strategy",
      llvm::cl::desc("Buffer allocation strategy"),
      llvm::cl::values(
        clEnumValN(AllocationStrategy::SaveBuffers, "save-buffers",
                   "Allocate first available buffer (default, minimize buffer count)"),
        clEnumValN(AllocationStrategy::BalancedUtilization, "balanced-utilization",
                   "Balance buffer utilization and reduce adjacent operation conflicts")
      ),
      llvm::cl::init(AllocationStrategy::SaveBuffers)};
  
  void runOnOperation() override;
  
private:
  /// Hardware configuration (can be set via pass options)
  PYNQHardwareConfig hwConfig;
  
  /// Process a single function
  LogicalResult processFunction(func::FuncOp funcOp);
  
  /// Transform buffer_request ops to buffer_alloc ops based on analysis
  LogicalResult applyAllocations(func::FuncOp funcOp,
                                BufferAllocationAnalysis &analysis,
                                SmallVector<LiveSegment> &segments);
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
  
  // First version: require a single block.
  if (funcOp.getBody().getBlocks().size() != 1) {
    singleBlock = nullptr;
    return;
  }

  singleBlock = &funcOp.getBody().front();
  opToIndex.clear();
  indexToOp.clear();

  unsigned index = 0;
  for (Operation &op : singleBlock->getOperations()) {
    opToIndex[&op] = index++;
    indexToOp.push_back(&op);
  }
}

std::optional<std::pair<unsigned, unsigned>>
BufferAllocationAnalysis::findFirstLastUseInBlock(Value value) {
  if (!value || !singleBlock)
    return std::nullopt;

  unsigned first = std::numeric_limits<unsigned>::max();
  unsigned last = 0;
  bool any = false;

  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (!user)
      continue;
    if (user->getBlock() != singleBlock)
      return std::nullopt;
    auto it = opToIndex.find(user);
    if (it == opToIndex.end())
      continue;
    any = true;
    first = std::min(first, it->second);
    last = std::max(last, it->second);
  }

  if (!any)
    return std::nullopt;
  return std::make_pair(first, last);
}

LogicalResult BufferAllocationAnalysis::collectSegments() {
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
  
  segments.clear();

  if (!singleBlock)
    return failure();

  funcOp.walk([&](BufferAllocOp allocOp) {
    if (!allocOp.isVirtual())
      return;

    Value buf = allocOp.getBuffer();
    auto firstLast = findFirstLastUseInBlock(buf);
    if (!firstLast.has_value()) {
      // Unused or used outside block: handled by caller.
      return;
    }

    LiveSegment seg;
    seg.virtualAlloc = allocOp;
    seg.virtualBuffer = buf;
    seg.startIndex = firstLast->first;
    seg.endIndex = firstLast->second;
    seg.requiredSizeBytes = allocOp.getCapacityBytes();
    seg.elementType = allocOp.getElementType();
    if (allocOp.hasRole())
      seg.role = allocOp.getRole();
    segments.push_back(seg);
  });
  
  return success();
}

LogicalResult BufferAllocationAnalysis::analyze() {
  buildLinearOrder();
  if (!singleBlock)
    return failure();
  return collectSegments();
}

//===----------------------------------------------------------------------===//
// LinearScanAllocator Implementation
//===----------------------------------------------------------------------===//

void LinearScanAllocator::expireOld(unsigned currentIndex) {
  for (auto it = active.begin(); it != active.end();) {
    LiveSegment *seg = *it;
    if (seg && seg->endIndex < currentIndex) {
      if (seg->assignedBufferId) {
        unsigned bufferId = *seg->assignedBufferId;
        freeBuffer(bufferId);
        // 记录 buffer 释放位置，用于 balanced-utilization 策略
        lastReleaseIndex[bufferId] = currentIndex;
      }
      it = active.erase(it);
      continue;
    }
    ++it;
  }
}

std::optional<unsigned> LinearScanAllocator::tryAllocate(LiveSegment &segment) {
  /*
   * 策略分发：根据配置选择不同的分配策略
   */
  
  switch (config.strategy) {
  case AllocationStrategy::SaveBuffers:
    return tryAllocateSaveBuffers(segment);
  case AllocationStrategy::BalancedUtilization:
    return tryAllocateBalancedUtilization(segment);
  }
  
  return std::nullopt;
}

std::optional<unsigned> LinearScanAllocator::tryAllocateSaveBuffers(LiveSegment &segment) {
  /*
   * SaveBuffers 策略（原始行为）:
   * 1. 找到第一个空闲的 buffer ID
   * 2. 检查该 buffer 的容量是否满足需求
   * 3. 如果满足，分配并返回 buffer ID
   * 4. 如果没有空闲 buffer，返回 nullopt
   * 
   * 目标：最小化使用的 buffer 数量
   */
  
  for (unsigned i = 0; i < config.numBuffers; ++i) {
    if (freeBuffers[i]) {
      if (segment.requiredSizeBytes <= config.defaultBufferCapacity) {
        allocateBuffer(i, segment);
        return i;
      }
    }
  }
  
  return std::nullopt;
}

std::optional<unsigned> LinearScanAllocator::tryAllocateBalancedUtilization(LiveSegment &segment) {
  /*
   * BalancedUtilization 策略:
   * 1. 尽量用满所有 buffer（负载均衡）
   * 2. 避免与相邻操作的 buffer 立即复用（减少伪依赖）
   * 3. 不增加 spill 风险（仍然只从空闲 buffer 中选择）
   * 
   * 评分规则：
   * - 非相邻释放（lastReleaseIndex != currentIndex-1）: +1000
   * - 使用次数少: -usageCount * 10
   * - ID 小作为 tie-breaker（稳定性）
   */
  
  // 检查容量是否满足
  if (segment.requiredSizeBytes > config.defaultBufferCapacity)
    return std::nullopt;
  
  int bestId = -1;
  int bestScore = std::numeric_limits<int>::min();
  
  for (unsigned i = 0; i < config.numBuffers; ++i) {
    if (!freeBuffers[i])
      continue;
    
    // 评分
    int score = 0;
    
    // 优先：避免刚在上一个 index 释放的 buffer（减少相邻冲突）
    if (lastReleaseIndex[i] + 1 != segment.startIndex)
      score += 1000;
    
    // 次要：负载均衡，优先使用历史使用次数少的 buffer
    score -= bufferUsageCount[i] * 10;
    
    // 选择最佳或更小 ID（稳定性）
    if (score > bestScore || (score == bestScore && (bestId < 0 || i < static_cast<unsigned>(bestId)))) {
      bestId = i;
      bestScore = score;
    }
  }
  
  if (bestId >= 0) {
    allocateBuffer(static_cast<unsigned>(bestId), segment);
    return static_cast<unsigned>(bestId);
  }
  
  return std::nullopt;
}

LiveSegment *LinearScanAllocator::selectSpillVictim(
    Operation *currentOp, ArrayRef<Value> currentOpBuffers) const {
  // Do not spill a segment if the current op uses that buffer.
  auto isUsedByCurrent = [&](LiveSegment *seg) {
    if (!seg)
      return false;
    for (Value b : currentOpBuffers)
      if (b == seg->virtualBuffer)
        return true;
    return false;
  };

  LiveSegment *victim = nullptr;
  unsigned farthestEnd = 0;

  for (LiveSegment *seg : active) {
    if (!seg || !seg->assignedBufferId)
      continue;
    if (isUsedByCurrent(seg))
      continue;
    // Heuristic: spill the one with farthest end.
    if (!victim || seg->endIndex > farthestEnd) {
      victim = seg;
      farthestEnd = seg->endIndex;
    }
  }

  (void)currentOp;
  return victim;
}

void LinearScanAllocator::freeBuffer(unsigned bufferId) {
  freeBuffers.set(bufferId);
}

void LinearScanAllocator::allocateBuffer(unsigned bufferId,
                                        LiveSegment &segment) {
  freeBuffers.reset(bufferId);
  segment.assignedBufferId = bufferId;
  active.push_back(&segment);
  // 更新使用计数，用于 balanced-utilization 策略
  bufferUsageCount[bufferId]++;
}

static SmallVector<Value, 8> collectPynqBufferOperands(Operation *op) {
  SmallVector<Value, 8> bufs;
  if (!op)
    return bufs;
  llvm::SmallPtrSet<Value, 8> seen;
  for (Value operand : op->getOperands()) {
    if (!operand)
      continue;
    if (!llvm::isa<pynq::BufferType>(operand.getType()))
      continue;
    if (seen.insert(operand).second)
      bufs.push_back(operand);
  }
  return bufs;
}

static std::optional<unsigned>
findNextUseIndex(Value v, unsigned fromIndex,
                 const BufferAllocationAnalysis &analysis) {
  unsigned best = std::numeric_limits<unsigned>::max();
  bool any = false;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!user)
      continue;
    auto idx = analysis.getIndexOf(user);
    if (!idx.has_value())
      continue;
    if (*idx < fromIndex)
      continue;
    any = true;
    best = std::min(best, *idx);
  }
  if (!any)
    return std::nullopt;
  return best;
}

LogicalResult LinearScanAllocator::allocate(
    SmallVector<LiveSegment> &segments,
    const BufferAllocationAnalysis &analysis) {
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
  llvm::sort(segments, [](const LiveSegment &a, const LiveSegment &b) {
    return a.startIndex < b.startIndex;
  });
  
  // We may append new segments during spilling; use index-based loop.
  for (size_t i = 0; i < segments.size(); ++i) {
    LiveSegment &seg = segments[i];
    expireOld(seg.startIndex);

    auto bufferId = tryAllocate(seg);
    if (bufferId)
      continue;

    if (!config.enableSpilling) {
      seg.virtualAlloc.emitError()
          << "failed to allocate buffer: all " << config.numBuffers
          << " buffers are occupied and spilling is disabled";
      return failure();
    }

    Operation *currentOp = analysis.getOpAt(seg.startIndex);
    auto currentOpBuffers = collectPynqBufferOperands(currentOp);
    LiveSegment *victim = selectSpillVictim(currentOp, currentOpBuffers);
    if (!victim || !victim->assignedBufferId) {
      seg.virtualAlloc.emitError() << "failed to spill any active buffer";
      return failure();
    }

    // Spill victim at seg.startIndex.
    auto nextUse = findNextUseIndex(victim->virtualBuffer, seg.startIndex,
                                   analysis);
    if (!nextUse.has_value()) {
      // Victim has no future uses; it can be simply truncated.
      victim->endIndex = seg.startIndex - 1;
    } else {
      victim->needsStoreToSlot = true;
      victim->storeBeforeIndex = seg.startIndex;

      LiveSegment reloadSeg;
      reloadSeg.virtualAlloc = victim->virtualAlloc;
      reloadSeg.virtualBuffer = victim->virtualBuffer;
      reloadSeg.startIndex = *nextUse;
      reloadSeg.endIndex = victim->endIndex;
      reloadSeg.requiredSizeBytes = victim->requiredSizeBytes;
      reloadSeg.elementType = victim->elementType;
      reloadSeg.role = victim->role;
      reloadSeg.needsReloadFromSlot = true;

      victim->endIndex = seg.startIndex - 1;

      segments.push_back(reloadSeg);
    }

    unsigned freedId = *victim->assignedBufferId;
    // Remove victim from active and free its buffer.
    for (auto it = active.begin(); it != active.end(); ++it) {
      if (*it == victim) {
        active.erase(it);
        break;
      }
    }
    freeBuffer(freedId);
    // 记录 spill 释放位置
    lastReleaseIndex[freedId] = seg.startIndex;

    bufferId = tryAllocate(seg);
    if (!bufferId) {
      seg.virtualAlloc.emitError() << "allocation still failed after spill";
      return failure();
    }

    // Re-sort the tail if we appended a reload segment.
    if (i + 1 < segments.size()) {
      llvm::sort(segments.begin() + (i + 1), segments.end(),
                 [](const LiveSegment &a, const LiveSegment &b) {
                   return a.startIndex < b.startIndex;
                 });
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

  // Wire options.
  hwConfig.numBuffers = numBuffers;
  hwConfig.defaultBufferCapacity = bufferCapacity;
  hwConfig.enableSpilling = enableSpilling;
  hwConfig.strategy = allocationStrategy;
  
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
  
  // First version is block-local: require a single block.
  if (funcOp.getBody().getBlocks().size() != 1) {
    return funcOp.emitError()
           << "pynq-buffer-allocation currently requires single-block functions";
  }

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
  
  BufferAllocationAnalysis analysis(funcOp, hwConfig);
  if (failed(analysis.analyze()))
    return funcOp.emitError() << "block-local liveness analysis failed";

  SmallVector<LiveSegment> segments = analysis.getSegments();
  if (segments.empty()) {
    // Erase unused virtual allocs to reduce noise.
    llvm::SmallVector<BufferAllocOp, 16> dead;
    funcOp.walk([&](BufferAllocOp allocOp) {
      if (allocOp.isVirtual() && allocOp.getBuffer().use_empty())
        dead.push_back(allocOp);
    });
    for (auto a : dead)
      a.erase();
    return success();
  }

  LinearScanAllocator allocator(hwConfig, analysis);
  if (failed(allocator.allocate(segments, analysis)))
    return failure();

  return applyAllocations(funcOp, analysis, segments);
}

LogicalResult PYNQBufferAllocationPass::applyAllocations(
    func::FuncOp funcOp,
    BufferAllocationAnalysis &analysis,
    SmallVector<LiveSegment> &segments) {
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
  
  Block *block = analysis.getSingleBlock();
  if (!block)
    return funcOp.emitError() << "expected single block";

  OpBuilder builder(funcOp.getContext());

  static constexpr StringLiteral kSpillSlotAttrName =
      "allo.pynq.spill_slot";

  // Determine which virtual buffers actually need spill slots.
  llvm::DenseMap<Value, bool> needsSlot;
  for (LiveSegment &seg : segments) {
    if (seg.needsStoreToSlot || seg.needsReloadFromSlot)
      needsSlot[seg.virtualBuffer] = true;
  }

  // Create spill slots near their first required insertion point.
  llvm::DenseMap<Value, Value> spillSlot;
  llvm::DenseMap<Value, unsigned> spillSlotFirstIndex;
  for (LiveSegment &seg : segments) {
    if (!(seg.needsStoreToSlot || seg.needsReloadFromSlot))
      continue;
    unsigned idx = seg.needsStoreToSlot ? seg.storeBeforeIndex : seg.startIndex;
    auto it = spillSlotFirstIndex.find(seg.virtualBuffer);
    if (it == spillSlotFirstIndex.end()) {
      spillSlotFirstIndex[seg.virtualBuffer] = idx;
    } else {
      it->second = std::min(it->second, idx);
    }
  }

  for (auto &it : needsSlot) {
    if (!it.second)
      continue;
    Value vbuf = it.first;
    auto bufTy = llvm::dyn_cast<pynq::BufferType>(vbuf.getType());
    if (!bufTy)
      continue;
    Type elemTy = bufTy.getElementType();
    int64_t capBytes = bufTy.getCapacityBytes();
    int64_t elemBits = elemTy.getIntOrFloatBitWidth();
    if (elemBits == 0 || (elemBits % 8) != 0)
      return funcOp.emitError() << "unsupported element type for spill slot";
    int64_t elemBytes = elemBits / 8;
    if (capBytes % elemBytes != 0)
      return funcOp.emitError() << "buffer capacity is not element-aligned";
    int64_t numElems = capBytes / elemBytes;

    auto slotType = MemRefType::get({numElems}, elemTy);

    unsigned insertIndex = 0;
    auto idxIt = spillSlotFirstIndex.find(vbuf);
    if (idxIt != spillSlotFirstIndex.end())
      insertIndex = idxIt->second;
    Operation *before = analysis.getOpAt(insertIndex);
    if (before)
      builder.setInsertionPoint(before);
    else
      builder.setInsertionPointToStart(block);

    auto alloc = builder.create<memref::AllocOp>(funcOp.getLoc(), slotType);
    alloc->setAttr(kSpillSlotAttrName, builder.getUnitAttr());
    spillSlot[vbuf] = alloc;
  }

  // Create physical buffer_alloc ops for each segment near the segment start.
  for (LiveSegment &seg : segments) {
    if (!seg.assignedBufferId)
      return seg.virtualAlloc.emitError() << "segment was not allocated";

    Operation *before = analysis.getOpAt(seg.startIndex);
    if (before)
      builder.setInsertionPoint(before);
    else
      builder.setInsertionPointToStart(block);

    auto newAlloc = builder.create<BufferAllocOp>(
        seg.virtualAlloc.getLoc(), seg.elementType, *seg.assignedBufferId,
        seg.requiredSizeBytes);
    if (seg.virtualAlloc.hasRole())
      newAlloc.setRoleAttr(seg.virtualAlloc.getRoleAttr());
    seg.physicalAlloc = newAlloc;
  }

  // Rewrite uses: for each segment, redirect uses in [start,end] to the
  // segment's physical buffer value.
  for (LiveSegment &seg : segments) {
    Value oldBuf = seg.virtualBuffer;
    Value newBuf = seg.physicalAlloc.getBuffer();
    for (OpOperand &use : llvm::make_early_inc_range(oldBuf.getUses())) {
      Operation *user = use.getOwner();
      auto idx = analysis.getIndexOf(user);
      if (!idx.has_value())
        continue;
      if (*idx < seg.startIndex || *idx > seg.endIndex)
        continue;
      use.set(newBuf);
    }
  }

  // Insert spill stores / reloads.
  for (LiveSegment &seg : segments) {
    Value slot = spillSlot.lookup(seg.virtualBuffer);

    if (seg.needsStoreToSlot) {
      Operation *before = analysis.getOpAt(seg.storeBeforeIndex);
      if (!before)
        return seg.virtualAlloc.emitError() << "invalid spill store point";
      if (!slot)
        return seg.virtualAlloc.emitError() << "missing spill slot";
      builder.setInsertionPoint(before);
      builder.create<pynq::CopyOp>(before->getLoc(),
                                  seg.physicalAlloc.getBuffer(), slot);
    }

    if (seg.needsReloadFromSlot) {
      Operation *before = analysis.getOpAt(seg.startIndex);
      if (!before)
        return seg.virtualAlloc.emitError() << "invalid reload point";
      if (!slot)
        return seg.virtualAlloc.emitError() << "missing spill slot";
      builder.setInsertionPoint(before);
      builder.create<pynq::CopyOp>(before->getLoc(), slot,
                                  seg.physicalAlloc.getBuffer());
    }
  }

  // Erase original virtual buffer_alloc ops (some may be unused).
  llvm::SmallVector<BufferAllocOp, 64> toErase;
  funcOp.walk([&](BufferAllocOp allocOp) {
    if (allocOp.isVirtual())
      toErase.push_back(allocOp);
  });
  for (auto a : toErase)
    a.erase();
  
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
