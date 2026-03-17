/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/PYNQConfig.h"
#include "allo/Dialect/PYNQOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

struct PackingConfig {
  int signBits;
  int signOffset;
  int rshiftBits;
  int rshiftOffset;
  int coeBits;
  int coeOffset;
  std::string coeMode;

  static PackingConfig parse(StringRef attr, StringRef mode) {
    SmallVector<int, 6> values;
    SmallVector<StringRef> pieces;
    attr.split(pieces, ',');
    if (pieces.size() != 6) {
      return {0, 22, 6, 16, 16, 0, mode.str()};
    }

    for (StringRef p : pieces) {
      int v = 0;
      if (p.getAsInteger(10, v)) {
        return {0, 22, 6, 16, 16, 0, mode.str()};
      }
      values.push_back(v);
    }

    return {values[0], values[1], values[2], values[3], values[4], values[5],
            mode.str()};
  }
};

struct UnpackedScale {
  int8_t sign;
  int16_t rshift;
  uint32_t coe;
};

static UnpackedScale unpackScale(uint32_t packed, const PackingConfig &config) {
  UnpackedScale result;

  uint32_t rshiftMask = (1u << config.rshiftBits) - 1u;
  result.rshift = static_cast<int16_t>((packed >> config.rshiftOffset) & rshiftMask);

  uint32_t coeMask = (1u << config.coeBits) - 1u;
  uint32_t coeRaw = (packed >> config.coeOffset) & coeMask;

  if (config.signBits > 0) {
    uint32_t signMask = (1u << config.signBits) - 1u;
    uint32_t signVal = (packed >> config.signOffset) & signMask;

    if (config.signBits == 1) {
      result.sign = (signVal == 0) ? 1 : -1;
    } else {
      result.sign = (signVal == 1) ? 1 : -1;
    }

    if (config.coeMode == "Tail") {
      result.coe = (1u << config.coeBits) | coeRaw;
    } else {
      result.coe = coeRaw;
    }
  } else {
    bool isNegative = ((coeRaw >> (config.coeBits - 1)) & 0x1u) != 0;
    result.sign = isNegative ? -1 : 1;
    if (isNegative) {
      result.coe = (~coeRaw + 1u) & coeMask;
    } else {
      result.coe = coeRaw;
    }
  }

  return result;
}

static uint32_t packScale(const UnpackedScale &unpacked,
                          const PackingConfig &config) {
  uint32_t packed = 0;

  if (config.signBits > 0) {
    uint32_t signVal = 0;
    if (config.signBits == 1) {
      signVal = (unpacked.sign < 0) ? 1u : 0u;
    } else {
      signVal = (unpacked.sign < 0) ? static_cast<uint32_t>(-1) : 1u;
    }
    uint32_t signMask = (1u << config.signBits) - 1u;
    packed |= ((signVal & signMask) << config.signOffset);
  }

  uint32_t rshiftMask = (1u << config.rshiftBits) - 1u;
  packed |= ((static_cast<uint32_t>(unpacked.rshift) & rshiftMask)
             << config.rshiftOffset);

  uint32_t coeToPack = 0;
  if (config.signBits > 0) {
    if (config.coeMode == "Tail") {
      uint32_t leadingBitPos = static_cast<uint32_t>(config.coeBits);
      coeToPack = unpacked.coe & ((1u << leadingBitPos) - 1u);
    } else {
      coeToPack = unpacked.coe;
    }
  } else {
    if (unpacked.sign < 0) {
      uint32_t mask = (1u << config.coeBits) - 1u;
      coeToPack = (~(unpacked.coe - 1u)) & mask;
    } else {
      coeToPack = unpacked.coe;
    }
  }

  uint32_t coeMask = (1u << config.coeBits) - 1u;
  packed |= ((coeToPack & coeMask) << config.coeOffset);
  return packed;
}

static LogicalResult adjustGlobalRshift(memref::GlobalOp globalOp,
                                        const PackingConfig &targetConfig,
                                        int delta) {
  auto initialValue = globalOp.getInitialValue();
  if (!initialValue)
    return failure();

  auto denseAttr = initialValue->dyn_cast<DenseElementsAttr>();
  if (!denseAttr)
    return failure();

  auto memrefTy = globalOp.getType().dyn_cast<MemRefType>();
  if (!memrefTy || !memrefTy.getElementType().isInteger(32))
    return failure();

  SmallVector<uint32_t> adjusted;
  adjusted.reserve(denseAttr.getNumElements());

  for (uint32_t val : denseAttr.getValues<uint32_t>()) {
    UnpackedScale unpacked = unpackScale(val, targetConfig);
    unpacked.rshift = static_cast<int16_t>(unpacked.rshift + delta);
    adjusted.push_back(packScale(unpacked, targetConfig));
  }

  auto newAttr = DenseElementsAttr::get(
      RankedTensorType::get(memrefTy.getShape(), IntegerType::get(globalOp.getContext(), 32)),
      llvm::ArrayRef(adjusted));
  globalOp.setInitialValueAttr(newAttr);
  return success();
}

static LogicalResult adjustScaleBinding(Value scale,
                                        Operation *user,
                                        StringRef bindingName,
                                        int desiredOffset,
                                        ModuleOp module,
                                        const PackingConfig &targetConfig,
                                        llvm::DenseMap<Operation *, int> &seenGlobals) {
  auto getGlobal = scale.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal) {
    user->emitError() << "expects " << bindingName
                      << " to be defined by memref.get_global";
    return failure();
  }

  auto globalOp = module.lookupSymbol<memref::GlobalOp>(getGlobal.getName());
  if (!globalOp) {
    user->emitError() << "cannot resolve global @" << getGlobal.getName()
                      << " for " << bindingName;
    return failure();
  }

  int delta = -desiredOffset; // desiredOffset - pynq::RShiftAdjustConfig::kInnateRshiftBias;

  auto it = seenGlobals.find(globalOp.getOperation());
  if (it != seenGlobals.end()) {
    if (it->second != delta) {
      user->emitError()
          << "global @" << globalOp.getName()
          << " is referenced by scales requiring different rshift delta ("
          << it->second << " vs " << delta << ")";
      return failure();
    }
    return success();
  }

  if (failed(adjustGlobalRshift(globalOp, targetConfig, delta))) {
    user->emitError() << "failed to adjust " << bindingName << " global @"
                      << globalOp.getName()
                      << " (expect i32 DenseElementsAttr packed by target config)";
    return failure();
  }

  seenGlobals[globalOp.getOperation()] = delta;
  return success();
}

class PYNQAdjustScaleRShiftPass
    : public PassWrapper<PYNQAdjustScaleRShiftPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PYNQAdjustScaleRShiftPass)

  StringRef getArgument() const final {
    return "pynq-adjust-scale-rshift";
  }

  StringRef getDescription() const final {
    return "Adjust packed-scale rshift for pynq ops based on per-op desired offset";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pynq::PYNQDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    PackingConfig targetConfig = PackingConfig::parse(
        pynq::PackedScaleConfig::kTargetPackingAttr,
        pynq::PackedScaleConfig::kTargetCoeModeAttr);

    llvm::DenseMap<Operation *, int> seenGlobals;

    module.walk([&](pynq::MatMulOp op) {
      if (failed(adjustScaleBinding(op.getFusedScale(), op, "fused_scale",
                                    pynq::RShiftAdjustConfig::kMatMulDropWidth,
                                    module, targetConfig, seenGlobals))) {
        signalPassFailure();
      }
    });

    module.walk([&](pynq::GELUOp op) {
      if (failed(adjustScaleBinding(op.getInScale(), op, "in_scale",
                                    pynq::VectorFixedPointConfig::kGELUInputFracBits,
                                    module, targetConfig, seenGlobals)) ||
          failed(adjustScaleBinding(op.getOutScaleInv(), op, "out_scale_inv",
                                    -pynq::VectorFixedPointConfig::kGELUOutputFracBits,
                                    module, targetConfig, seenGlobals))) {
        signalPassFailure();
      }
    });

    module.walk([&](pynq::SoftmaxOp op) {
      if (failed(adjustScaleBinding(op.getInScale(), op, "in_scale",
                                    pynq::VectorFixedPointConfig::kSoftmaxInputFracBits,
                                    module, targetConfig, seenGlobals)) ||
          failed(adjustScaleBinding(op.getOutScaleInv(), op, "out_scale_inv",
                                    -pynq::VectorFixedPointConfig::kSoftmaxExpFracBits,
                                    module, targetConfig, seenGlobals))) {
        signalPassFailure();
      }
    });

    module.walk([&](pynq::QAddOp op) {
      if (failed(adjustScaleBinding(op.getXScale(), op, "x_scale",
                                    pynq::VectorFixedPointConfig::kShortcutInputFracBits,
                                    module, targetConfig, seenGlobals)) ||
          failed(adjustScaleBinding(op.getYScale(), op, "y_scale",
                                    pynq::VectorFixedPointConfig::kShortcutInputFracBits,
                                    module, targetConfig, seenGlobals)) ||
          failed(adjustScaleBinding(op.getOScaleInv(), op, "o_scale_inv",
                                    -pynq::VectorFixedPointConfig::kShortcutOutputFracBits,
                                    module, targetConfig, seenGlobals))) {
        signalPassFailure();
      }
    });

    module.walk([&](pynq::LayerNormOp op) {
      if (failed(adjustScaleBinding(
              op.getFusedScale(), op, "fused_scale",
              -pynq::VectorFixedPointConfig::kLayerNormInvSqrtFracBits,
              module, targetConfig, seenGlobals))) {
        signalPassFailure();
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::allo::createPYNQAdjustScaleRShiftPass() {
  return std::make_unique<PYNQAdjustScaleRShiftPass>();
}
