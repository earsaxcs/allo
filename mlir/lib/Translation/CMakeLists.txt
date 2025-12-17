# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

add_mlir_library(MLIRAlloEmitHLSCpp
  ${globbed}

  DEPENDS
  MLIRPYNQOpsIncGen
  MLIRPYNQTypesIncGen
  MLIRPYNQAttrsIncGen
  MLIRPYNQEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRAllo
  MLIRAlloSupport
  MLIRMemRefDialect
  MLIRAnalysis
)