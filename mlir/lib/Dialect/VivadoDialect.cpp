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
  auto inputType = getInput().getType().dyn_cast<ShapedType>();
  if (!inputType)
    return emitOpError("input must be a shaped type (memref or tensor)");

  auto weightType = getWeight().getType().dyn_cast<ShapedType>();
  if (!weightType)
    return emitOpError("weight must be a shaped type (memref or tensor)");

  auto outputType = getOutput().getType().dyn_cast<ShapedType>();
  if (!outputType)
    return emitOpError("output must be a shaped type (memref or tensor)");

  if (!inputType.hasRank() || !weightType.hasRank() || !outputType.hasRank())
    return success();

  if (inputType.getRank() < 2)
    return emitOpError("input rank must be at least 2");
  if (outputType.getRank() < 2)
    return emitOpError("output rank must be at least 2");
  if (weightType.getRank() != 2)
    return emitOpError("weight rank must be 2");

  // Output rank should match input rank (batch dims preserved).
  if (outputType.getRank() != inputType.getRank())
    return emitOpError("output rank must match input rank");

  auto matchDim = [&](int64_t a, int64_t b) {
    return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
  };

  auto inShape = inputType.getShape();
  auto wShape = weightType.getShape();
  auto outShape = outputType.getShape();

  // Batch prefix dims must match between input and output.
  for (int64_t i = 0; i + 2 < inputType.getRank(); ++i) {
    if (!matchDim(inShape[i], outShape[i]))
      return emitOpError("batch dimensions mismatch between input and output");
  }

  bool transposeMode = getTransposeMode();
  bool outIsTransposed = getIsTransposed();

  // When transpose_mode is enabled but the output is kept in non-transposed layout,
  // require output scales to be rank-1 i32 memrefs, and (when statically known)
  // allow common quantization granularities:
  //   - per-tensor  : memref<1xi32>
  //   - per-token   : memref<Mxi32>
  //   - per-channel : memref<Nxi32>
  // where output layout is [..., M, N].
  if (transposeMode && !outIsTransposed) {
    auto requireSingletonScaleMemRef = [&](Value scale, StringRef which) -> LogicalResult {
      auto memrefTy = scale.getType().dyn_cast<MemRefType>();
      if (!memrefTy || memrefTy.getRank() != 1)
        return emitOpError(which) << " must be a memref<1xi32> when transpose_mode=true and is_transposed=false";
      auto shape = memrefTy.getShape();
      if (shape.size() != 1 || shape[0] != 1)
        return emitOpError(which) << " must be a memref<1xi32> when transpose_mode=true and is_transposed=false";
      if (!memrefTy.getElementType().isInteger(32))
        return emitOpError(which) << " must be a memref<1xi32> when transpose_mode=true and is_transposed=false";
      return success();
    };

    if (failed(requireSingletonScaleMemRef(getOscl(), "oscl (output_scale)")))
      return failure();
    if (failed(requireSingletonScaleMemRef(getOsclInv(), "oscl_inv (output_scale_inv)")))
      return failure();
  }

  // Shape conventions:
  // - transpose_mode=false: input [..., M, K] x weight [K, N] -> output [..., M, N]
  // - transpose_mode=true : input [..., K, M] x weight [K, N] ->
  //     output [..., M, N] when is_transposed=false
  //     output [..., N, M] when is_transposed=true
  if (!transposeMode) {
    int64_t M = inShape[inputType.getRank() - 2];
    int64_t K = inShape[inputType.getRank() - 1];
    if (!matchDim(wShape[1], K))
      return emitOpError("weight[1] must match input K dimension when transpose_mode=false");
    int64_t N = wShape[0];
    if (!matchDim(outShape[outputType.getRank() - 2], M) ||
        !matchDim(outShape[outputType.getRank() - 1], N))
      return emitOpError("output tail dims must be [M, N] when transpose_mode=false");
  } else {
    int64_t K = inShape[inputType.getRank() - 2];
    int64_t M = inShape[inputType.getRank() - 1];
    if (!matchDim(wShape[0], K))
      return emitOpError("weight[0] must match input K dimension when transpose_mode=true");
    int64_t N = wShape[1];

    if (!outIsTransposed) {
      if (!matchDim(outShape[outputType.getRank() - 2], M) ||
          !matchDim(outShape[outputType.getRank() - 1], N))
        return emitOpError(
            "output tail dims must be [M, N] when transpose_mode=true and is_transposed=false");
    } else {
      if (!matchDim(outShape[outputType.getRank() - 2], N) ||
          !matchDim(outShape[outputType.getRank() - 1], M))
        return emitOpError(
            "output tail dims must be [N, M] when transpose_mode=true and is_transposed=true");
    }
  }

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
   // So 3D bias can just appear after seperating bias pass and disable fused_bias
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

    // NOTE: input/weight/output shape checks are performed above.
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

  // Enforce RHS (y) scale to be a singleton memref for backend scale configuration.
  // NOTE: Only applies to y_scale (RHS). No restriction on x_scale (LHS).
  {
    auto yScaleTy = getYScale().getType().dyn_cast<MemRefType>();
    if (!yScaleTy || yScaleTy.getRank() != 1 || !yScaleTy.getElementType().isInteger(32))
      return emitOpError("y_scale must be memref<1xi32>");
    auto yShape = yScaleTy.getShape();
    if (yShape.size() != 1 || yShape[0] != 1)
      return emitOpError("y_scale must be memref<1xi32>");
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
