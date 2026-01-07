/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Dialect/VivadoDialect.h"
#include "allo/Dialect/VivadoOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::allo::vivado;

//===----------------------------------------------------------------------===//
// Vivado Dialect
//===----------------------------------------------------------------------===//

#include "allo/Dialect/VivadoDialect.cpp.inc"

void VivadoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "allo/Dialect/VivadoOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// QLinearOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult QLinearOp::verify() {
  auto outputType = getOutput().getType().dyn_cast<ShapedType>();
  if (!outputType)
    return emitOpError("output must be a shaped type (memref or tensor)");

  Value bias = getBias();
  if (!bias)
    return success();

  auto biasType = bias.getType().dyn_cast<ShapedType>();
  if (!biasType)
    return emitOpError("bias must be a shaped type (memref or tensor)");

  // Accept bias rank 1 or 2; compare against input's tail dims excluding batch.
  if (!outputType.hasRank())
    return success();

  if (outputType.getRank() < 2)
    return emitOpError("output rank must be at least 2 to compare with bias");

  if (biasType.getRank() < 1 || biasType.getRank() > 2)
    return emitOpError("bias rank must be 1 or 2 to align with input tail dims");

  if (outputType.hasStaticShape() && biasType.hasStaticShape()) {
    auto outShape = outputType.getShape();
    auto biasShape = biasType.getShape();

    // Drop batch dim (dim 0) from input for comparison.
    SmallVector<int64_t, 3> tail;
    for (size_t i = 1; i < outShape.size(); ++i)
      tail.push_back(outShape[i]);

    // Compare tail suffix of length biasRank with biasShape.
    if (tail.size() < static_cast<size_t>(biasType.getRank()))
      return emitOpError("bias rank exceeds output tail dimensions");

    for (int i = 0; i < biasType.getRank(); ++i) {
      int64_t expect = tail[tail.size() - biasType.getRank() + i];
      int64_t got = biasShape[i];
      if (got != ShapedType::kDynamic && expect != ShapedType::kDynamic && got != expect) {
        return emitOpError("bias shape ") << biasShape
               << " must match the last " << biasType.getRank()
               << " dims of output tail " << ArrayRef<int64_t>(tail);
      }
    }

    // TOOD: verify weight and input, but it's more complex due to transposes. so hang
    // is_transposed == false: input [..., IC] x weight [OC, IC] -> output [..., OC]
    // is_transposed == true:  input [..., IC, M] x weight [IC, OC] -> output [..., OC, M]
  }

  return success();
}

//===----------------------------------------------------------------------===//
// QMatMulOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult QMatMulOp::verify() {
  // Validate RHS (second input) layout attribute.
  // Supported: "bhld", "blhd", "bhdl". Default is "bhld".
  if (auto layoutAttr = (*this)->getAttrOfType<StringAttr>("rhs_layout")) {
    auto v = layoutAttr.getValue();
    if (!(v == "bhld" || v == "blhd" || v == "bhdl")) {
      return emitOpError("rhs_layout must be one of {bhld, blhd, bhdl}, got '")
             << v << "'";
    }
  }

  // Validate reduce_dim is within RHS rank range when rank is known.
  if (auto reduceAttr = (*this)->getAttrOfType<IntegerAttr>("reduce_dim")) {
    if (auto rhsType = getRhs().getType().dyn_cast<ShapedType>()) {
      if (rhsType.hasRank()) {
        int64_t rhsRank = rhsType.getRank();
        int64_t reduceDim = reduceAttr.getInt();
        if (reduceDim < 0 || reduceDim >= rhsRank) {
          return emitOpError("reduce_dim must be within [0, rhs_rank), got ")
                 << reduceDim << " for rhs_rank=" << rhsRank;
        }
      }
    }
  }

  // NOTE: Do not uncomment this.
  // auto lhsType = getLhs().getType().dyn_cast<ShapedType>();
  // auto rhsType = getRhs().getType().dyn_cast<ShapedType>();
  // if (!lhsType || !rhsType)
  //   return emitOpError("lhs and rhs must be shaped types (memref or tensor)");

  // if (!lhsType.hasRank() || !rhsType.hasRank())
  //   return success();

  // int64_t lhsRank = lhsType.getRank();
  // int64_t rhsRank = rhsType.getRank();
  // if (!((lhsRank == 2 && rhsRank == 2) || (lhsRank == 3 && rhsRank == 3) || (lhsRank == 4 && rhsRank == 4)))
  //   return emitOpError("lhs and rhs must both be rank-2 or both rank-3 or both rank-4");

  // auto matchDim = [&](int64_t a, int64_t b) {
  //   return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
  // };

  // bool transposed = getIsTransposed();
  // if (!transposed) {
  //   // Expect lhs last dim == rhs first non-batch dim.
  //   int64_t lhsReduce = lhsType.getShape()[lhsRank - 1];
  //   int64_t rhsReduce = rhsType.getShape()[rhsRank - 2];
  //   if (!matchDim(lhsReduce, rhsReduce))
  //     return emitOpError("lhs reduce dim ") << lhsReduce
  //            << " must match rhs non-batch first dim " << rhsReduce
  //            << " when is_transposed=false";
  // } else {
  //   // Expect lhs and rhs share the non-batch first dim (exposed feature dim).
  //   int64_t lhsFeat = lhsType.getShape()[lhsRank - 2];
  //   int64_t rhsFeat = rhsType.getShape()[rhsRank - 2];
  //   if (!matchDim(lhsFeat, rhsFeat))
  //     return emitOpError("lhs non-batch first dim ") << lhsFeat
  //            << " must match rhs non-batch first dim " << rhsFeat
  //            << " when is_transposed=true";
  // }

  return success();
}

//===----------------------------------------------------------------------===//
// QMatMulIsqrtDOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult QMatMulIsqrtDOp::verify() {
  // Validate RHS (second input) layout attribute.
  // Supported: "bhld", "blhd", "bhdl". Default is "bhld".
  if (auto layoutAttr = (*this)->getAttrOfType<StringAttr>("rhs_layout")) {
    auto v = layoutAttr.getValue();
    if (!(v == "bhld" || v == "blhd" || v == "bhdl")) {
      return emitOpError("rhs_layout must be one of {bhld, blhd, bhdl}, got '")
             << v << "'";
    }
  }

  // Validate reduce_dim is within RHS rank range when rank is known.
  if (auto reduceAttr = (*this)->getAttrOfType<IntegerAttr>("reduce_dim")) {
    if (auto rhsType = getRhs().getType().dyn_cast<ShapedType>()) {
      if (rhsType.hasRank()) {
        int64_t rhsRank = rhsType.getRank();
        int64_t reduceDim = reduceAttr.getInt();
        if (reduceDim < 0 || reduceDim >= rhsRank) {
          return emitOpError("reduce_dim must be within [0, rhs_rank), got ")
                 << reduceDim << " for rhs_rank=" << rhsRank;
        }
      }
    }
  }

  // NOTE: Do not uncomment this.
  // auto lhsType = getLhs().getType().dyn_cast<ShapedType>();
  // auto rhsType = getRhs().getType().dyn_cast<ShapedType>();
  // if (!lhsType || !rhsType)
  //   return emitOpError("lhs and rhs must be shaped types (memref or tensor)");

  // if (!lhsType.hasRank() || !rhsType.hasRank())
  //   return success();

  // int64_t lhsRank = lhsType.getRank();
  // int64_t rhsRank = rhsType.getRank();
  // if (!((lhsRank == 2 && rhsRank == 2) || (lhsRank == 3 && rhsRank == 3) || (lhsRank == 4 && rhsRank == 4)))
  //   return emitOpError("lhs and rhs must both be rank-2 or both rank-3 or both rank-4");

  // auto matchDim = [&](int64_t a, int64_t b) {
  //   return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
  // };

  // bool transposed = getIsTransposed();
  // if (!transposed) {
  //   // Expect lhs last dim == rhs first non-batch dim.
  //   int64_t lhsReduce = lhsType.getShape()[lhsRank - 1];
  //   int64_t rhsReduce = rhsType.getShape()[rhsRank - 2];
  //   if (!matchDim(lhsReduce, rhsReduce))
  //     return emitOpError("lhs reduce dim ") << lhsReduce
  //            << " must match rhs non-batch first dim " << rhsReduce
  //            << " when is_transposed=false";
  // } else {
  //   // Expect lhs and rhs share the non-batch first dim (exposed feature dim).
  //   int64_t lhsFeat = lhsType.getShape()[lhsRank - 2];
  //   int64_t rhsFeat = rhsType.getShape()[rhsRank - 2];
  //   if (!matchDim(lhsFeat, rhsFeat))
  //     return emitOpError("lhs non-batch first dim ") << lhsFeat
  //            << " must match rhs non-batch first dim " << rhsFeat
  //            << " when is_transposed=true";
  // }

  return success();
}

//===----------------------------------------------------------------------===//
// QAddOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult QAddOp::verify() {
  // Get lhs and rhs types
  auto lhsType = getLhs().getType().dyn_cast<ShapedType>();
  auto rhsType = getRhs().getType().dyn_cast<ShapedType>();
  auto outputType = getOutput().getType().dyn_cast<ShapedType>();
  
  if (!lhsType || !rhsType || !outputType) {
    return emitOpError("operands must be shaped types (memref or tensor)");
  }
  
  // Check that lhs and rhs have the same shape
  if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
    if (lhsType.getShape() != rhsType.getShape()) {
      return emitOpError("lhs shape ")
             << lhsType.getShape() << " does not match rhs shape "
             << rhsType.getShape() << ". "
             << "For bias addition from QLinear, ensure bias is broadcasted "
             << "to match the output shape before creating QAddOp. "
             << "Hint: bias shape should be [1, 1, OC] or [B, L, OC] "
             << "to match QLinear output [B, L, OC].";
    }
  }
  
  // Check that output matches lhs/rhs shape
  if (outputType.hasStaticShape() && lhsType.hasStaticShape()) {
    if (outputType.getShape() != lhsType.getShape()) {
      return emitOpError("output shape ")
             << outputType.getShape() << " does not match operand shape "
             << lhsType.getShape();
    }
  }
  
  return success();
}

#define GET_OP_CLASSES
#include "allo/Dialect/VivadoOps.cpp.inc"
