# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_dialect(Core core)
add_mlir_doc(Core Core Core/ -gen-dialect-doc -dialect=core)

set(LLVM_TARGET_DEFINITIONS Core.td)

mlir_tablegen(CoreAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CoreAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRCoreAttributesIncGen)
