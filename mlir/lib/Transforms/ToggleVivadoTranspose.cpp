/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// ToggleVivadoTransposePass
// 
// This pass transforms the IR from standard layout (B, L, D) to transposed 
// layout (B, D, L) for hardware-friendly execution.
//
// Key transformations:
// 1. Transpose weight globals: (OC, IC) -> (IC, OC) for QLinear
// 2. Transpose bias globals: (L, OC) -> (OC, L) for QLinear (if 2D)
// 3. Toggle transpose_mode attribute on all vivado quant ops
// 4. Update activation shapes, reshape ops, and linalg.transpose
// 5. Insert boundary transposes: vivado.activation_layout_transpose at entry/exit
//
// This pass only supports False -> True transformation (one direction).
//===----------------------------------------------------------------------===//

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/PYNQConfig.h"
#include "allo/Dialect/VivadoOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG false
#define DEBUG2 false

using namespace mlir;
using namespace allo;

namespace vivado_ops = mlir::allo::vivado;

namespace mlir {
namespace allo {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static memref::GlobalOp resolveGetGlobal(ModuleOp module, Value value) {
  Value cur = value;
  for (int depth = 0; depth < 16 && cur; ++depth) {
    if (auto getGlobal = cur.getDefiningOp<memref::GetGlobalOp>()) {
      if (auto globalOp = module.lookupSymbol<memref::GlobalOp>(getGlobal.getName()))
        return globalOp;
      return {};
    }

    Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      return {};

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      cur = subviewOp.getSource();
      continue;
    }

    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(defOp)) {
      cur = reshapeOp.getSource();
      continue;
    }

    return {};
  }
  return {};
}

/// Transpose a 2D DenseElementsAttr
static DenseElementsAttr transposeMatrix(DenseElementsAttr attr, 
                                          ArrayRef<int64_t> shape) {
  if (shape.size() != 2) {
    return attr; // Only handle 2D
  }
  
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  auto elementType = attr.getElementType();
  
  // Create transposed shape
  SmallVector<int64_t, 2> newShape = {cols, rows};
  auto newTensorType = RankedTensorType::get(newShape, elementType);
  
  // For integer types (weights are typically i8)
  // If bias is 2D, it's 8-bit to adapt to activations.
  if (elementType.isInteger(8)) {
    auto values = attr.getValues<int8_t>();
    SmallVector<int8_t> transposed(rows * cols);
    
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        // Original: data[i][j] = values[i * cols + j]
        // Transposed: new_data[j][i] = values[i * cols + j]
        transposed[j * rows + i] = *(values.begin() + i * cols + j);
      }
    }
    
    return DenseElementsAttr::get(newTensorType, llvm::ArrayRef(transposed));
  }
  
  // For i32 (maybe bias or scales)
  // If bias is 1D, it's 32-bit to broadcast and filled into fifo.
  if (elementType.isInteger(32)) {
    auto values = attr.getValues<int32_t>();
    SmallVector<int32_t> transposed(rows * cols);
    
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        transposed[j * rows + i] = *(values.begin() + i * cols + j);
      }
    }
    
    return DenseElementsAttr::get(newTensorType, llvm::ArrayRef(transposed));
  }
  
  // For f32
  if (elementType.isF32()) {
    auto values = attr.getValues<float>();
    SmallVector<float> transposed(rows * cols);
    
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        transposed[j * rows + i] = *(values.begin() + i * cols + j);
      }
    }
    
    return DenseElementsAttr::get(newTensorType, llvm::ArrayRef(transposed));
  }
  
  // Unsupported type, return original
  return attr;
}

/// Check if all vivado quant ops have transpose_mode = false
static bool verifyAllNotTransposed(ModuleOp module) {
  bool allFalse = true;
  
  module.walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<BoolAttr>("transpose_mode")) {
      if (attr.getValue()) {
        allFalse = false;
      }
    }
  });
  
  return allFalse;
}

/// Get all vivado quant ops that have transpose_mode attribute
/// For QLinear ops, filter by layer_type:
///   - proj_q, proj_k, out_proj, fc1, fc2: include
///   - proj_v: exclude (no transpose needed)
static void collectVivadoQuantOps(ModuleOp module, 
                                   SmallVector<Operation*> &ops) {
  module.walk([&](Operation *op) {
    // Check if op is a vivado dialect op with transpose_mode attribute
    if (op->getDialect() && 
        op->getDialect()->getNamespace() == "vivado" &&
        op->hasAttr("transpose_mode") && !op->getAttr("transpose_mode").cast<BoolAttr>().getValue()) {
      
      // Special handling for QLinear ops based on layer_type
      if (isa<vivado_ops::QLinearOp>(op)) {
        auto layerTypeAttr = op->getAttrOfType<StringAttr>("layer_type");
        if (layerTypeAttr) {
          StringRef layerType = layerTypeAttr.getValue();
          // proj_v and classifier.dense should NOT be transposed
          // but you can't just skip it here because we need to set transpose_mode = true later
          // if (pynq::QLinearLayerType::isQKVGemmProjV(layerType) || pynq::QLinearLayerType::isClassifierDense(layerType)) {
          //   return; // Skip this op
          // }

          // proj_q, proj_k, out_proj, fc1, fc2 should be transposed
          // (all other types are included by default)
        }
      }
      
      ops.push_back(op);
    }
  });
}

/// Check if a value (memref) is connected to vivado quant ops through use-def chain
/// This recursively traces through memref operations like alloc, subview, reshape, etc.
static bool isConnectedToQuantOp(Value value, int maxDepth = 10) {
  llvm::DenseSet<const void *> visited;

  std::function<bool(Value, int)> impl = [&](Value v, int depth) -> bool {
    if (!v)
      return false;
    if (depth <= 0)
      return false;

    const void *key = v.getAsOpaquePointer();
    if (visited.contains(key))
      return false;
    visited.insert(key);

    auto isDirectVivadoQuant = [&](Operation *op) -> bool {
      return op && op->getDialect() && op->getDialect()->getNamespace() == "vivado" &&
             op->hasAttr("transpose_mode");
    };

    // --- Use chain (destination-passing aware) ---
    for (Operation *user : v.getUsers()) {
      if (isDirectVivadoQuant(user))
        return true;

      if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
        if (v == subviewOp.getSource()) {
          if (impl(subviewOp.getResult(), depth - 1))
            return true;
        }
        continue;
      }

      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(user)) {
        if (v == reshapeOp.getSource()) {
          if (impl(reshapeOp.getResult(), depth - 1))
            return true;
        }
        continue;
      }

      if (auto transposeOp = dyn_cast<linalg::TransposeOp>(user)) {
        // linalg.transpose is destination-passing style on buffers.
        if (v == transposeOp.getInput()) {
          if (impl(transposeOp.getInit(), depth - 1))
            return true;
        } else if (v == transposeOp.getInit()) {
          if (impl(transposeOp.getInput(), depth - 1))
            return true;
        }
        continue;
      }
    }

    // --- Def chain ---
    if (Operation *defOp = v.getDefiningOp()) {
      if (isDirectVivadoQuant(defOp))
        return true;

      if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp))
        return impl(subviewOp.getSource(), depth - 1);
      if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(defOp))
        return impl(reshapeOp.getSource(), depth - 1);
      if (auto transposeOp = dyn_cast<linalg::TransposeOp>(defOp))
        return impl(transposeOp.getInput(), depth - 1);
    }

    return false;
  };

  return impl(value, maxDepth);
}

/// Trace back to find the source quant op type for a value
/// Returns: "qlinear.proj_q", "qlinear.proj_k", "qlinear.proj_v", 
///          "qmatmul", "qadd", "other_quant", "non_quant"
static std::string classifyVivadoQuantOpType(Operation *op) {
  if (!op)
    return "non_quant";

  if (isa<vivado_ops::QLinearOp>(op)) {
    auto layerTypeAttr = op->getAttrOfType<StringAttr>("layer_type");
    if (layerTypeAttr) {
      StringRef layerType = layerTypeAttr.getValue();
      if (pynq::QLinearLayerType::isQKVGemmProjQ(layerType))
        return "qlinear.proj_q";
      if (pynq::QLinearLayerType::isQKVGemmProjK(layerType))
        return "qlinear.proj_k";
      if (pynq::QLinearLayerType::isQKVGemmProjV(layerType))
        return "qlinear.proj_v";
      if (pynq::QLinearLayerType::isClassifierDense(layerType))
        return "qlinear.classifier";
      return "qlinear.other";
    }
    return "qlinear.other";
  }

  if (isa<vivado_ops::QMatMulOp>(op) || isa<vivado_ops::QMatMulIsqrtDOp>(op))
    return "qmatmul";

  if (isa<vivado_ops::QAddOp>(op))
    return "qadd";

  if (isa<vivado_ops::DequantOp>(op))
    return "dequant";

  return "other_quant";
}

static std::string getSourceQuantOpTypeImpl(Value value,
                                            llvm::DenseSet<const void *> &visited,
                                            Operation *fatherOp,
                                            int maxDepth,
                                            Operation **userOp = nullptr) {
  if (!value)
    return "non_quant";
  if (maxDepth <= 0)
    return "non_quant";

  const void *key = value.getAsOpaquePointer();
  if (visited.contains(key))
    return "non_quant";
  visited.insert(key);

  auto tryClassifyVivado = [&](Operation *op) -> std::string {
    if (!op)
      return "non_quant";
    if (op->getDialect() && op->getDialect()->getNamespace() == "vivado" &&
        op->hasAttr("transpose_mode")) {
      return classifyVivadoQuantOpType(op);
    }
    return "non_quant";
  };

  // 0) Check first if this value is connected to a vivado quant op through use-def chain
  // NOTE: This method can't solve the case where multiple vivado quant ops uses `value`
  // In Transformer this case is usually not possible. But it worth noting.
  for (Operation *user : value.getUsers()) {
    if (DEBUG) {
      llvm::errs() << "==============================================================\n";
      llvm::errs() << "user: " << user->getName().getStringRef().str() << "\n";
      user->dump();
      if (user->getNumOperands() > 0)
        user->getOperand(0).dump();
      value.dump();
      llvm::errs() << (user && user->getDialect() && user->getDialect()->getNamespace() == "vivado" &&
          user->hasAttr("transpose_mode") && user->getNumOperands() > 0 &&
          user->getOperand(0) == value) << "\n";
      llvm::errs() << "fatherOp: " << "\n";
      if (fatherOp) {
        fatherOp->dump();
      }
      else {
        llvm::errs() << "null\n";
      }
      llvm::errs() << "==============================================================\n";
    }

    if (user && user->getDialect() && user->getDialect()->getNamespace() == "vivado" &&
        user->hasAttr("transpose_mode") && user->getNumOperands() > 0 &&
        user->getOperand(0) == value) {
      if (userOp) {
        *userOp = user;
      }
      return classifyVivadoQuantOpType(user);
    }
  }

  // 1) Walk backward through defining ops.
  if (Operation *defOp = value.getDefiningOp()) {
    // if (auto direct = tryClassifyVivado(defOp); direct != "non_quant")
    //   return direct;

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp); subviewOp && defOp != fatherOp) {
      return getSourceQuantOpTypeImpl(subviewOp.getSource(), visited, defOp, maxDepth - 1, userOp);
    }

    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(defOp); reshapeOp && defOp != fatherOp) {
      return getSourceQuantOpTypeImpl(reshapeOp.getSource(), visited, defOp, maxDepth - 1, userOp);
    }

    // if (auto transposeOp = dyn_cast<linalg::TransposeOp>(defOp)) {
    //   // Note: linalg.transpose is destination-passing; for buffer form it does not
    //   // produce SSA memref results. If we're looking at a result value (tensor form),
    //   // follow the input.
    //   return getSourceQuantOpTypeImpl(transposeOp.getInput(), visited, transposeOp, maxDepth - 1);
    // }
  }

  // 2) Walk forward through users (important for destination-passing buffers).
  for (Operation *user : value.getUsers()) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(user); subviewOp && user != fatherOp) {
      // memref.subview(%source, ...) -> %result
      if (value == subviewOp.getSource())
        return getSourceQuantOpTypeImpl(subviewOp.getResult(), visited, user, maxDepth - 1, userOp);
      continue;
    }

    if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(user); reshapeOp && user != fatherOp) {
      // memref.reshape %source(%shape) -> %result
      if (value == reshapeOp.getSource())
        return getSourceQuantOpTypeImpl(reshapeOp.getResult(), visited, user, maxDepth - 1, userOp);
      continue;
    }

    if (auto transposeOp = dyn_cast<linalg::TransposeOp>(user); transposeOp && user != fatherOp) {
      // Destination-passing buffer semantics:
      // - If current value is the input, the data flows into the init buffer.
      // - If current value is the init buffer, its contents are sourced from input.
      if (value == transposeOp.getInput())
        return getSourceQuantOpTypeImpl(transposeOp.getInit(), visited, user, maxDepth - 1, userOp);
      if (value == transposeOp.getInit())
        return getSourceQuantOpTypeImpl(transposeOp.getInput(), visited, user, maxDepth - 1, userOp);
      continue;
    }
  }

  return "non_quant";
}

static std::string getSourceQuantOpType(Value value, Operation* fatherOp, int maxDepth = 10, Operation **userOp = nullptr) {
  llvm::DenseSet<const void *> visited;
  if (DEBUG)
    llvm::errs() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
  return getSourceQuantOpTypeImpl(value, visited, fatherOp, maxDepth, userOp);
  if (DEBUG)
    llvm::errs() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << "\n";
}

//===----------------------------------------------------------------------===//
// Core Transformation Functions
//===----------------------------------------------------------------------===//

using namespace pynq;

//===--------------------------------------------------------------------===//
// Step 1: Transpose weight and bias globals
//===--------------------------------------------------------------------===//

static LogicalResult transposeWeightGlobals(ModuleOp module) {
  llvm::DenseSet<Operation *> seen;
  SmallVector<memref::GlobalOp> toTranspose;

  // Collect only globals referenced by QLinear ops that we actually transpose.
  module.walk([&](vivado_ops::QLinearOp qlinearOp) {
    auto layerTypeAttr = qlinearOp->getAttrOfType<StringAttr>("layer_type");
    StringRef layerType = layerTypeAttr ? layerTypeAttr.getValue() : "unknown";

    // Weight must be transposed
    if (auto w = resolveGetGlobal(module, qlinearOp.getWeight())) {
      if (seen.insert(w.getOperation()).second)
        toTranspose.push_back(w);
    }

    // Bias is transposed according to its layer type
    if (qlinearOp.getBias()) {
      if (auto b = resolveGetGlobal(module, qlinearOp.getBias())) {
        if (!QLinearLayerType::isQKVGemmProjV(layerType) && 
            !QLinearLayerType::isClassifierDense(layerType)) {
          // proj_v and classifier.dense does NOT transpose bias
          if (seen.insert(b.getOperation()).second)
            toTranspose.push_back(b);
        }
      }
    }
  });
  
  // Transpose each weight/bias global
  for (auto globalOp : toTranspose) {
    auto memrefType = globalOp.getType().cast<MemRefType>();
    auto shape = memrefType.getShape();
    
    // Only transpose 2D tensors
    if (shape.size() != 2)
      continue;
    
    // Get initial value
    auto initialValue = globalOp.getInitialValue();
    if (!initialValue)
      continue;
    
    auto denseAttr = initialValue->dyn_cast<DenseElementsAttr>();
    if (!denseAttr)
      continue;
    
    // Transpose the data
    auto transposedAttr = transposeMatrix(denseAttr, shape);
    
    // Create new type with transposed shape: (M, N) -> (N, M)
    // For weight: (OC, IC) -> (IC, OC)
    // For bias: (L, OC) -> (OC, L)
    auto newType = MemRefType::get({shape[1], shape[0]}, 
                                    memrefType.getElementType());
    
    // Update global
    globalOp.setType(newType);
    globalOp.setInitialValueAttr(transposedAttr);
    
    // Update all GetGlobalOp users to use the new type
    auto symbolName = globalOp.getSymName();
    module.walk([&](memref::GetGlobalOp getGlobal) {
      if (getGlobal.getName() == symbolName) {
        getGlobal.getResult().setType(newType);
      }
    });
  }
  
  return success();
}

//===--------------------------------------------------------------------===//
// Step 3: Toggle transpose_mode and update shapes (AFTER inserting boundaries)
// Now that boundaries are marked, shape updates won't affect non-quant regions
//===--------------------------------------------------------------------====//

/// Update activation shapes for intermediate buffers between vivado ops
/// Changes 3D memref types from (B,L,D) to (B,D,L)
static bool updateIntermediateActivationShapes(ModuleOp module, MLIRContext *context) {
  bool ok = true;
  // NOTE: Before use this function, make sure all quant ops that need to be toggled are updated (their mark)
  auto shouldTransposeActivation = [&](Value value, Operation* fatherOp) -> bool {
    Operation *op = nullptr;
    std::string sourceOpType = getSourceQuantOpType(value, fatherOp, 10, &op);
    if (sourceOpType == "non_quant")
      return false;
    // 不再只判断type
    // if (sourceOpType == "qlinear.proj_v")
    //   return false;
    // if (sourceOpType == "qlinear.classifier")
    //   return false;
    // if (sourceOpType == "dequant")
    //   return false;
    if (op->getAttr("transpose_mode").cast<BoolAttr>().getValue()) {
      return op->getAttr("is_transposed").cast<BoolAttr>().getValue();
    }
    return true;
  };

  // Step 1: Update all vivado quant ops' result types
  // NOTE: Vivado Ops are not SSA, so here actually not affects
  SmallVector<Operation*> quantOps;
  collectVivadoQuantOps(module, quantOps);

  auto setRhsLayoutAttr = [&](Operation *op, Value rhs) {
    auto setLayoutAndReduceDim = [&](StringRef layout, int32_t reduceDim) {
      op->setAttr("rhs_layout", StringAttr::get(context, layout));
      op->setAttr("reduce_dim",
                  IntegerAttr::get(IntegerType::get(context, 32), reduceDim));
    };

    std::string rhsSource = getSourceQuantOpType(rhs, op);
    if (rhsSource == "qlinear.proj_k") {
      // proj_k (K): when used as RHS of attn.QK matmul, reduction is over head-dim `d`.
      // Under rhs_layout="bhdl" (B,H,D,L), the d axis is 2.
      setLayoutAndReduceDim("bhdl", /*reduceDim=*/2);
      return;
    }
    if (rhsSource == "qlinear.proj_v") {
      // proj_v (V): when used as RHS of attn.SV matmul, reduction is over token-length `l`.
      // Under rhs_layout="blhd" (B,L,H,D), the l axis is 1.
      setLayoutAndReduceDim("blhd", /*reduceDim=*/1);
      return;
    }
    op->emitError() << "Unsupported RHS source type for rhs_layout: " << rhsSource
                    << ". Expected qlinear.proj_k or qlinear.proj_v.";
    ok = false;
  };
  
  // NOTE: actually no use for vivado ops since they are not SSA
  // auto transposeResults = [&](Operation *op) {
  //   for (auto result : op->getResults()) {
  //     auto memrefType = result.getType().dyn_cast<MemRefType>();
  //     if (memrefType && memrefType.getRank() == 3) {
  //       auto shape = memrefType.getShape();
  //       SmallVector<int64_t> newShape = {shape[0], shape[2], shape[1]};
  //       auto newType = MemRefType::get(newShape, memrefType.getElementType());
  //       result.setType(newType);
  //     }
  //     else if (memrefType && memrefType.getRank() == 2) {
  //       auto shape = memrefType.getShape();
  //       SmallVector<int64_t> newShape = {shape[1], shape[0]};
  //       auto newType = MemRefType::get(newShape, memrefType.getElementType());
  //       result.setType(newType);
  //     }
  //     else if (memrefType && memrefType.getRank() == 4) {
  //       auto shape = memrefType.getShape();
  //       SmallVector<int64_t> newShape = {shape[0], shape[1], shape[3], shape[2]};
  //       auto newType = MemRefType::get(newShape, memrefType.getElementType());
  //       result.setType(newType);
  //     }
  //   }
  // };

  // Step 1: Update all quant ops' result types and transpose attributes
  for (auto *op : quantOps) {
    // For qmatmul/qmatmul_isqrtd, tag RHS layout for backend lowering.
    if (auto qmm = dyn_cast<vivado_ops::QMatMulOp>(op)) {
      setRhsLayoutAttr(op, qmm.getRhs());
    } else if (auto qmmIsqrt = dyn_cast<vivado_ops::QMatMulIsqrtDOp>(op)) {
      setRhsLayoutAttr(op, qmmIsqrt.getRhs());
    }

    // Special case: when toggling transpose mode, move Softmax reduction axis to
    // the second-to-last dimension (i.e., dim=2 for rank=4:[B,H,L,L]).
    if (auto intSoftmax = dyn_cast<vivado_ops::IntSoftmaxOp>(op)) {
      auto rank = intSoftmax.getOutput().getType().cast<MemRefType>().getRank();
      op->setAttr("axis", IntegerAttr::get(IntegerType::get(context, 64), rank - 2));
    }

    if (auto qlinear = dyn_cast<vivado_ops::QLinearOp>(op)) {
      auto layerTypeAttr = qlinear->getAttrOfType<StringAttr>("layer_type");
      StringRef layerType = layerTypeAttr ? layerTypeAttr.getValue() : "unknown";
      // proj_v and classifier dense 不转置，其余（q/k/out/fc 等或者单一测试的线性层）转置
      op->setAttr("transpose_mode", BoolAttr::get(context, true));
      if (!pynq::QLinearLayerType::isQKVGemmProjV(layerType) && !pynq::QLinearLayerType::isClassifierDense(layerType)) {
        // transposeResults(op);
        op->setAttr("is_transposed", BoolAttr::get(context, true));
      }
      continue;
    }

    // Dequant和其前置量化op的转置模式一致
    if (isa<vivado_ops::DequantOp>(op)) {
      op->setAttr("transpose_mode", BoolAttr::get(context, true));
      // op->setAttr("is_transposed", BoolAttr::get(context, false));
      Operation *srcOp = nullptr;
      std::string sourceOpType = getSourceQuantOpType(
        dyn_cast<vivado_ops::DequantOp>(op).getInput(), op, 10, &srcOp);
      if (sourceOpType != "non_quant" && srcOp->hasAttr("transpose_mode")) {
        op->setAttr("is_transposed", BoolAttr::get(context, srcOp->getAttr("is_transposed").cast<BoolAttr>().getValue()));
      }
      continue;
    }

    // 其它 vivado quant op：统一转置激活形状并置 transpose_mode 和 is_transposed
    // transposeResults(op);
    op->setAttr("transpose_mode", BoolAttr::get(context, true));
    op->setAttr("is_transposed", BoolAttr::get(context, true));
  }
  
  // Step 2: Update memref.alloc ops that create activation buffers
  if (DEBUG)
    llvm::errs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Updating memref.alloc ops !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  module.walk([&](memref::AllocOp allocOp) {
    auto memrefType = allocOp.getType().dyn_cast<MemRefType>();
    if (!memrefType || (memrefType.getRank() != 3 && memrefType.getRank() != 2 && memrefType.getRank() != 4))
      return;

    if (DEBUG2) {
      allocOp.dump();
      llvm::errs() << "Source op type: " << getSourceQuantOpType(allocOp.getResult(), allocOp.getOperation()) << "\n";
    }

    // NOTE: Some alloc may not be transposed（some linalg.transpose's result） 
    // but fortunately it will be deleted not used later
    if (!shouldTransposeActivation(allocOp.getResult(), allocOp.getOperation()))
      return;
    
    // Transpose the shape: (B, L, D) -> (B, D, L) or (B, D) -> (D, B) or (B, H, L, D) -> (B, H, D, L)
    auto shape = memrefType.getShape();
    SmallVector<int64_t> newShape;
    if (memrefType.getRank() == 3) {
      newShape = {shape[0], shape[2], shape[1]};
    } else if (memrefType.getRank() == 2) {
      newShape = {shape[1], shape[0]};
    }
    else if (memrefType.getRank() == 4) {
      newShape = {shape[0], shape[1], shape[3], shape[2]};
    }
    auto newType = MemRefType::get(newShape, memrefType.getElementType());
    
    // Update the alloc op's result type
    allocOp.getResult().setType(newType);
  });
  
  // Step 3: Update memref.subview ops that extract views from activation buffers
  if (DEBUG)
    llvm::errs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Updating memref.subview ops !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  module.walk([&](memref::SubViewOp subviewOp) {
    auto sourceType = subviewOp.getSource().getType().dyn_cast<MemRefType>();
    auto resultType = subviewOp.getType().dyn_cast<MemRefType>();
    
    if (!sourceType || !resultType)
      return;
    
    if (!shouldTransposeActivation(subviewOp.getSource(), subviewOp.getOperation()))
      return;
    
    // Transpose the result type
    if (resultType.getRank() == 3) {
      auto shape = resultType.getShape();
      SmallVector<int64_t> newShape = {shape[0], shape[2], shape[1]};
      auto newType = MemRefType::get(newShape, resultType.getElementType(),
                                      resultType.getLayout(), 
                                      resultType.getMemorySpace());
      subviewOp.getResult().setType(newType);
    } else if (resultType.getRank() == 2) {
      auto shape = resultType.getShape();
      SmallVector<int64_t> newShape = {shape[1], shape[0]};
      auto newType = MemRefType::get(newShape, resultType.getElementType(),
                                      resultType.getLayout(), 
                                      resultType.getMemorySpace());
      subviewOp.getResult().setType(newType);
    } else if (resultType.getRank() == 4) {
      auto shape = resultType.getShape();
      SmallVector<int64_t> newShape = {shape[0], shape[1], shape[3], shape[2]};
      auto newType = MemRefType::get(newShape, resultType.getElementType(),
                                      resultType.getLayout(), 
                                      resultType.getMemorySpace());
      subviewOp.getResult().setType(newType);
    }
  });
  
  // Step 4: Update memref.reshape ops (for multi-head attention split)
  if (DEBUG) 
    llvm::errs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Updating memref.reshape ops !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  module.walk([&](memref::ReshapeOp reshapeOp) {
    auto sourceType = reshapeOp.getSource().getType().dyn_cast<MemRefType>();
    auto resultType = reshapeOp.getType().dyn_cast<MemRefType>();
    
    if (!sourceType || !resultType)
      return;
    
    // Check if the reshape is connected to vivado quant ops through use-def chain
    bool connectedToQuantOp = isConnectedToQuantOp(reshapeOp.getSource()) /*|| 
                               isConnectedToQuantOp(reshapeOp.getResult())*/;
    
    if (!connectedToQuantOp)
      return; // Not in quant region, skip
    
    // Get source op type
    std::string sourceOpType = getSourceQuantOpType(reshapeOp.getSource(), reshapeOp.getOperation());
    if (DEBUG2) {
      reshapeOp.dump();
      llvm::errs() << "Source op type: " << sourceOpType << "\n";
    }
    
    // Special handling based on source op type
    bool shouldTranspose = false;
    if (sourceOpType == "qlinear.proj_q" || sourceOpType == "qlinear.proj_k" || sourceOpType == "qlinear.other") {
      // proj_q/proj_k: transpose output
      shouldTranspose = true;
    } else if (sourceOpType == "qlinear.proj_v" || sourceOpType == "classifier.dense") {
      // proj_v: no transpose
      shouldTranspose = false;
    } else if (sourceOpType == "qmatmul") {
      // qmatmul: transpose output
      shouldTranspose = true;
    } else if (sourceOpType == "non_quant") {
      // Not in quant region
      return;
    } else {
      // Unimplemented case
      reshapeOp.emitError() << "Unimplemented reshape source type: " << sourceOpType;
      return;
    }
    
    // Apply transformation if needed
    if (shouldTranspose && resultType.getRank() == 4) {
      auto shape = resultType.getShape();
      // Original: (B, L, D) -> (B, L, H, D/H)
      // Transposed: (B, D, L) -> (B, H, D/H, L)
      SmallVector<int64_t> newShape = {shape[0], shape[2], shape[3], shape[1]};
      auto newType = MemRefType::get(newShape, resultType.getElementType());
      reshapeOp.getResult().setType(newType);
      
      // Update shape operand if it's from a global constant
      auto shapeOp = reshapeOp.getShape().getDefiningOp();
      if (auto getGlobal = dyn_cast_or_null<memref::GetGlobalOp>(shapeOp)) {
        auto globalName = getGlobal.getName();
        module.walk([&](memref::GlobalOp globalOp) {
          if (globalOp.getSymName() == globalName) {
            auto globalType = globalOp.getType().cast<MemRefType>();
            // Update the shape data in the global
            if (auto initialValue = globalOp.getInitialValue()) {
              if (auto denseAttr = initialValue->dyn_cast<DenseElementsAttr>()) {
                SmallVector<int64_t> newShapeData(newShape.begin(), newShape.end());
                auto newAttr = DenseElementsAttr::get(
                    RankedTensorType::get({static_cast<int64_t>(newShapeData.size())}, 
                                          globalType.getElementType()),
                    llvm::ArrayRef(newShapeData));
                globalOp.setInitialValueAttr(newAttr);
              }
            }
          }
        });
      }
    }
    else if (shouldTranspose && resultType.getRank() == 3) {
      auto shape = resultType.getShape();
      // Original: (B, L, H, D // H) -> (B, L, D)
      // Transposed: (B, H, D // H, L) -> (B, D, L) 
      SmallVector<int64_t> newShape = {shape[0], shape[2], shape[1]};
      auto newType = MemRefType::get(newShape, resultType.getElementType());
      reshapeOp.getResult().setType(newType);
      
      // Update shape operand if it's from a global constant
      auto shapeOp = reshapeOp.getShape().getDefiningOp();
      if (auto getGlobal = dyn_cast_or_null<memref::GetGlobalOp>(shapeOp)) {
        auto globalName = getGlobal.getName();
        module.walk([&](memref::GlobalOp globalOp) {
          if (globalOp.getSymName() == globalName) {
            auto globalType = globalOp.getType().cast<MemRefType>();
            // Update the shape data in the global
            if (auto initialValue = globalOp.getInitialValue()) {
              if (auto denseAttr = initialValue->dyn_cast<DenseElementsAttr>()) {
                SmallVector<int64_t> newShapeData(newShape.begin(), newShape.end());
                auto newAttr = DenseElementsAttr::get(
                    RankedTensorType::get({static_cast<int64_t>(newShapeData.size())}, 
                                          globalType.getElementType()),
                    llvm::ArrayRef(newShapeData));
                globalOp.setInitialValueAttr(newAttr);
              }
            }
          }
        });
      }
    }
  });
  
  // Step 5: Update linalg.transpose ops in quant op region
  if (DEBUG)
    llvm::errs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Updating linalg.transpose ops !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  SmallVector<Operation*> transposesToDelete;
  module.walk([&](linalg::TransposeOp transposeOp) {
    auto inputType = transposeOp.getInput().getType().dyn_cast<ShapedType>();
    auto outputType = transposeOp.getInit().getType().dyn_cast<ShapedType>();
    
    if (!inputType || !outputType)
      return;
    
    // Check if this transpose is connected to quant ops through use-def chain
    bool connectedToQuantOp = isConnectedToQuantOp(transposeOp.getInput()) /*|| 
                               isConnectedToQuantOp(transposeOp.getInit())*/;
    
    if (!connectedToQuantOp)
      return; // Not in quant region, skip (no change)
    
    // Get source op type
    std::string sourceOpType = getSourceQuantOpType(transposeOp.getInput(), transposeOp.getOperation());
    
    if (DEBUG2) {
      transposeOp.dump();
      llvm::errs() << "Source op type: " << sourceOpType << "\n";
    }

    // Special handling based on source op type
    if (sourceOpType == "qlinear.proj_q" || sourceOpType == "qlinear.proj_k" || sourceOpType == "qlinear.other") {
      // proj_q/proj_k: directly remove transpose
      transposesToDelete.push_back(transposeOp);
      return;
    } else if (sourceOpType == "qlinear.proj_v") {
      // proj_v: also remove transpose
      transposesToDelete.push_back(transposeOp);
      return;
    } else if (sourceOpType == "qmatmul") { // Note this is not qmatmul_isqrtd
      // qmatmul: remove transpose
      transposesToDelete.push_back(transposeOp);
      return;
    } else if (sourceOpType == "non_quant") {
      // Not in quant region (shouldn't reach here due to connectedToQuantOp check)
      return;
    } else {
      // Unimplemented case
      transposeOp.emitError() << "Unimplemented transpose source type: " << sourceOpType;
      ok = false;
      return;
    }
    
    // NOTE: The old generic permutation adjustment logic is removed.
    // Now we handle transposes with specific rules based on source op type (above).
    // The code below should not be reached if source type identification works correctly.
    /*
    // Fallback: old generic logic (should not reach here)
    auto permutation = transposeOp.getPermutation();
    SmallVector<int64_t> permVec(permutation.begin(), permutation.end());
    
    // Apply transpose transformation to permutation
    // For 3D: swap dimensions 1 and 2 in the permutation
    // For 4D: swap dimensions 2 and 3 in the permutation
    if (permVec.size() == 3) {
      // Original permutation operates on (B, L, D)
      // After layout change to (B, D, L), we need to adjust permutation
      // If original was [0, 1, 2] (identity), after transpose it should be [0, 2, 1]
      // If original was [0, 2, 1] (transpose L,D), after transpose it should be [0, 1, 2] (identity)
      SmallVector<int64_t> newPerm(3);
      for (size_t i = 0; i < 3; ++i) {
        int64_t mappedDim = permVec[i];
        if (mappedDim == 1) mappedDim = 2;
        else if (mappedDim == 2) mappedDim = 1;
        newPerm[i] = mappedDim;
      }
      
      // Check if new permutation is identity
      bool isIdentity = true;
      for (size_t i = 0; i < newPerm.size(); ++i) {
        if (newPerm[i] != static_cast<int64_t>(i)) {
          isIdentity = false;
          break;
        }
      }
      
      if (isIdentity) {
        // Mark for deletion
        transposesToDelete.push_back(transposeOp);
      } else {
        // Update permutation
        transposeOp.setPermutation(newPerm);
      }
    } else if (permVec.size() == 4) {
      // For 4D: (B, H, L, D/H) -> (B, H, D/H, L)
      // Adjust permutation by swapping positions 2 and 3
      SmallVector<int64_t> newPerm(4);
      for (size_t i = 0; i < 4; ++i) {
        int64_t mappedDim = permVec[i];
        if (mappedDim == 2) mappedDim = 3;
        else if (mappedDim == 3) mappedDim = 2;
        newPerm[i] = mappedDim;
      }
      
      // Check if new permutation is identity
      bool isIdentity = true;
      for (size_t i = 0; i < newPerm.size(); ++i) {
        if (newPerm[i] != static_cast<int64_t>(i)) {
          isIdentity = false;
          break;
        }
      }
      
      if (isIdentity) {
        // Mark for deletion
        transposesToDelete.push_back(transposeOp);
      } else {
        // Update permutation
        transposeOp.setPermutation(newPerm);
      }
    }
      */
  });
  
  // llvm::errs() << "Number of transposes to delete: " << transposesToDelete.size() << "\n";
  // for (auto *op : transposesToDelete) {
  //   // Replace uses of output with input
  //   op->dump();
  // }
  // return;
  
  // Delete identity transpose ops
  for (auto *op : transposesToDelete) {
    // Replace uses of output with input
    // op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    op->getOperand(1).replaceAllUsesWith(op->getOperand(0));
    op->erase();
  }

  return ok;
}

static bool toggleAllQuantOps(ModuleOp module, MLIRContext *context) {
  // Update intermediate activation buffer shapes
  return updateIntermediateActivationShapes(module, context);
}

//===--------------------------------------------------------------------===//
// Step 2: Insert boundary transposes (BEFORE toggling attributes)
// This establishes clear boundaries for the quantization region
//===--------------------------------------------------------------------====//

static LogicalResult insertBoundaryTransposes(ModuleOp module, MLIRContext *context) {
  // Find all functions that contain vivado quant ops
  SmallVector<func::FuncOp> funcsWithQuantOps;
  
  module.walk([&](func::FuncOp funcOp) {
    bool hasQuantOp = false;
    funcOp.walk([&](Operation *op) {
      if (op->getDialect() && 
          op->getDialect()->getNamespace() == "vivado" &&
          op->hasAttr("transpose_mode")) {
        hasQuantOp = true;
      }
    });
    if (hasQuantOp) {
      funcsWithQuantOps.push_back(funcOp);
    }
  });
  
  // For each function, find entry and exit points for transpose insertion
  for (auto funcOp : funcsWithQuantOps) {
    // Find first quant ops: vivado.quant (quantization op, entry point)
    SmallVector<Operation*> firstQuantOps;
    funcOp.walk([&](Operation *op) {
      if (isa<vivado_ops::QuantOp>(op)) {
        firstQuantOps.push_back(op);
      }
    });
    
    // Find last quant ops: vivado.dequant (dequantization op, exit point)
    SmallVector<Operation*> lastQuantOps;
    funcOp.walk([&](Operation *op) {
      if (isa<vivado_ops::DequantOp>(op)) {
        lastQuantOps.push_back(op);
      }
    });
    
    if (firstQuantOps.empty() || lastQuantOps.empty())
      continue;
    
    // Insert entry transposes for all first quant ops
    for (auto *firstQuantOp : firstQuantOps) {
      OpBuilder builder(firstQuantOp);
      
      // For vivado.quant: operand 0 is output buffer, operand 1 is input (float tensor)
      // We need to transpose the input before quantization
      Value activationInput;
      if (firstQuantOp->getNumOperands() > 1) {
        activationInput = firstQuantOp->getOperand(1);  // Input float tensor
      }
      
      if (activationInput) {
        auto inputType = activationInput.getType().cast<MemRefType>();
        auto shape = inputType.getShape();
        
        // Determine shape transformation based on rank
        SmallVector<int64_t> transposedShape;
        if (shape.size() == 3) {
          // Standard layout (B, L, D) -> Transposed layout (B, D, L)
          transposedShape = {shape[0], shape[2], shape[1]};
        } else if (shape.size() == 2) {
          // Standard layout (B, D) -> Transposed layout (D, B)
          transposedShape = {shape[1], shape[0]};
        // } else if (shape.size() == 4) {
        //   // deprecated for begin and end boundary transpose
        //   // Standard layout (B, H, L, D/H) -> Transposed layout (B, H, D/H, L)
        //   transposedShape = {shape[0], shape[1], shape[3], shape[2]};
        } else {
          // Unsupported rank, skip
          continue;
        }
        
        auto transposedType = MemRefType::get(transposedShape, 
                                               inputType.getElementType());
        
        // Allocate buffer for transposed activation
        auto allocOp = builder.create<memref::AllocOp>(
            firstQuantOp->getLoc(), transposedType);
        
        // Create entry transpose op: to_transposed=true means (B,L,D)->(B,D,L)
        auto entryTranspose = builder.create<vivado_ops::ActivationLayoutTransposeOp>(
            firstQuantOp->getLoc(),
            allocOp.getResult(),  // output
            activationInput,      // input
            builder.getBoolAttr(true));  // to_transposed=true
        
        // Replace the activation input in firstQuantOp with transposed buffer
        for (auto &operand : firstQuantOp->getOpOperands()) {
          if (operand.get() == activationInput) {
            operand.set(allocOp.getResult());
          }
        }
        
        // Keep marker attribute
        firstQuantOp->setAttr("_needs_entry_transpose", 
                             BoolAttr::get(context, true));
      }
    }
    
    // Insert exit transposes for all last quant ops
    for (auto *lastQuantOp : lastQuantOps) {
      OpBuilder builder(lastQuantOp);
      builder.setInsertionPointAfter(lastQuantOp);
      
      // Find activation output buffer (first operand is output for destination-passing style)
      // NOTE: Vivado ops don't produce SSA results, they write to pre-allocated buffers
      Value activationOutput;
      if (lastQuantOp->getNumOperands() > 0) {
        auto firstOperand = lastQuantOp->getOperand(0);
        auto memrefType = firstOperand.getType().dyn_cast<MemRefType>();
        if (memrefType && (memrefType.getRank() == 3 || memrefType.getRank() == 2 || memrefType.getRank() == 4)) {
          activationOutput = firstOperand;
        }
      }
      
      if (activationOutput) {
        auto outputType = activationOutput.getType().cast<MemRefType>();
        auto shape = outputType.getShape();
        
        // Determine shape transformation based on rank
        SmallVector<int64_t> originalShape;
        
        // NOTE: For exit, we will transpose it in the last qlinear classifier op, so just keep original shape
        // play a role as a border marker
        if (true) {
          originalShape.assign(shape.begin(), shape.end());
        } else if (shape.size() == 3) {
          // (B, L, D) -> will be transposed to (B, D, L) in quant region
          // So exit needs to convert (B, D, L) -> (B, L, D)
          originalShape = {shape[0], shape[2], shape[1]};
        } else if (shape.size() == 2) {
          // (B, D) -> will be transposed to (D, B)
          // So exit needs to convert (D, B) -> (B, D)
          originalShape = {shape[1], shape[0]};
        // } else if (shape.size() == 4) {
        //   // deprecated for begin and end boundary transpose
        //   // (B, H, L, D/H) -> will be transposed to (B, H, D/H, L)
        //   // So exit needs to convert (B, H, D/H, L) -> (B, H, L, D/H)
        //   originalShape = {shape[0], shape[1], shape[3], shape[2]};
        } else {
          originalShape.assign(shape.begin(), shape.end());
        }
        
        auto originalType = MemRefType::get(originalShape, 
                                            outputType.getElementType());
        
        // Allocate buffer for de-transposed activation
        auto allocOp = builder.create<memref::AllocOp>(
            lastQuantOp->getLoc(), originalType);
        
        // Create exit transpose op: to_transposed=false means transposed->standard layout
        auto exitTranspose = builder.create<vivado_ops::ActivationLayoutTransposeOp>(
            lastQuantOp->getLoc(),
            allocOp.getResult(),   // output
            activationOutput,      // input (the output buffer of lastQuantOp)
            builder.getBoolAttr(false));  // to_transposed=false
        
        // Replace all uses of activationOutput buffer (after lastQuantOp) with de-transposed buffer
        // We need to find users that come AFTER the quant region
        SmallVector<OpOperand*> usesToReplace;
        for (auto &use : activationOutput.getUses()) {
          auto *userOp = use.getOwner();
          // Skip if user is the quant op itself or the exit transpose
          if (userOp == lastQuantOp || userOp == exitTranspose) {
            continue;
          }
          // Skip if user is another vivado quant op (inside quant region)
          if (userOp->getDialect() && 
              userOp->getDialect()->getNamespace() == "vivado" &&
              userOp->hasAttr("transpose_mode")) {
            continue;
          }
          // This is an external user, needs de-transposed buffer
          usesToReplace.push_back(&use);
        }
        for (auto *use : usesToReplace) {
          use->set(allocOp.getResult());
        }
        
        // Keep marker attribute
        lastQuantOp->setAttr("_needs_exit_transpose", 
                            BoolAttr::get(context, true));
      }
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Main Transformation Entry Point
//===----------------------------------------------------------------------===//

bool applyToggleVivadoTranspose(ModuleOp &module, MLIRContext *context) {
  // Step 0: Verify precondition - all ops should be transpose_mode=false
  if (!verifyAllNotTransposed(module)) {
    module.emitError() << "ToggleVivadoTransposePass requires all vivado "
                       << "ops to have transpose_mode=false. "
                       << "This pass only supports False -> True transformation.";
    return false;
  }
  
  // Step 1: Transpose weight and bias globals (for QLinear)
  // Weight: (OC, IC) -> (IC, OC)
  // Bias (if 2D): (L, OC) -> (OC, L)
  if (failed(transposeWeightGlobals(module))) {
    return false;
  }
  
  // Step 2: Insert boundary transposes BEFORE toggling attributes
  // This ensures clear boundaries for quant region detection
  if (failed(insertBoundaryTransposes(module, context))) {
    return false;
  }
  
  // Step 3: Toggle transpose_mode on all vivado quant ops and update shapes
  // Now that boundaries are marked, shape updates won't affect non-quant regions
  if (!toggleAllQuantOps(module, context))
    return false;
  
  // Mark module as transposed
  module->setAttr("vivado.layout_transposed", 
                  BoolAttr::get(context, true));
  
  return true;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ToggleVivadoTransposePass
    : public PassWrapper<ToggleVivadoTransposePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToggleVivadoTransposePass)

  StringRef getArgument() const override { return "toggle-vivado-transpose"; }
  StringRef getDescription() const override {
    return "Toggle layout from (B,L,D) to (B,D,L) for Vivado backend";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vivado::VivadoDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    
    if (!applyToggleVivadoTranspose(module, context)) {
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createToggleVivadoTransposePass() {
  return std::make_unique<ToggleVivadoTransposePass>();
}

} // namespace allo
} // namespace mlir
