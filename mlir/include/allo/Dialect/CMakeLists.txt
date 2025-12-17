# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS AlloOps.td)
mlir_tablegen(AlloOps.h.inc -gen-op-decls)
mlir_tablegen(AlloOps.cpp.inc -gen-op-defs)
mlir_tablegen(AlloDialect.h.inc -gen-dialect-decls -dialect=allo)
mlir_tablegen(AlloDialect.cpp.inc -gen-dialect-defs -dialect=allo)
add_public_tablegen_target(MLIRAlloOpsIncGen)
add_dependencies(mlir-headers MLIRAlloOpsIncGen)

set(LLVM_TARGET_DEFINITIONS AlloTypes.td)
mlir_tablegen(AlloTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AlloTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRAlloTypesIncGen)

set(LLVM_TARGET_DEFINITIONS AlloAttrs.td)
mlir_tablegen(AlloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AlloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAlloAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS AlloAttrs.td)
mlir_tablegen(AlloEnums.h.inc -gen-enum-decls)
mlir_tablegen(AlloEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAlloEnumsIncGen)

add_mlir_doc(AlloDialect AlloDialect Allo/ -gen-dialect-doc)
add_mlir_doc(AlloOps AlloOps Allo/ -gen-op-doc)

# Vivado dialect
set(LLVM_TARGET_DEFINITIONS VivadoQuantOps.td)
mlir_tablegen(VivadoOps.h.inc -gen-op-decls)
mlir_tablegen(VivadoOps.cpp.inc -gen-op-defs)
mlir_tablegen(VivadoDialect.h.inc -gen-dialect-decls -dialect=vivado)
mlir_tablegen(VivadoDialect.cpp.inc -gen-dialect-defs -dialect=vivado)
add_public_tablegen_target(MLIRVivadoOpsIncGen)
add_dependencies(mlir-headers MLIRVivadoOpsIncGen)

add_mlir_doc(VivadoDialect VivadoDialect Vivado/ -gen-dialect-doc)
add_mlir_doc(VivadoQuantOps VivadoOps Vivado/ -gen-op-doc)

# PYNQ dialect
set(LLVM_TARGET_DEFINITIONS PYNQOps.td)
mlir_tablegen(PYNQOps.h.inc -gen-op-decls)
mlir_tablegen(PYNQOps.cpp.inc -gen-op-defs)
mlir_tablegen(PYNQDialect.h.inc -gen-dialect-decls -dialect=pynq)
mlir_tablegen(PYNQDialect.cpp.inc -gen-dialect-defs -dialect=pynq)
add_public_tablegen_target(MLIRPYNQOpsIncGen)
add_dependencies(mlir-headers MLIRPYNQOpsIncGen)

set(LLVM_TARGET_DEFINITIONS PYNQTypes.td)
mlir_tablegen(PYNQTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PYNQTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRPYNQTypesIncGen)
add_dependencies(mlir-headers MLIRPYNQTypesIncGen)

set(LLVM_TARGET_DEFINITIONS PYNQAttrs.td)
mlir_tablegen(PYNQAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(PYNQAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRPYNQAttrsIncGen)
add_dependencies(mlir-headers MLIRPYNQAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS PYNQAttrs.td)
mlir_tablegen(PYNQEnums.h.inc -gen-enum-decls)
mlir_tablegen(PYNQEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPYNQEnumsIncGen)
add_dependencies(mlir-headers MLIRPYNQEnumsIncGen)

add_mlir_doc(PYNQDialect PYNQDialect PYNQ/ -gen-dialect-doc)
add_mlir_doc(PYNQOps PYNQOps PYNQ/ -gen-op-doc)
