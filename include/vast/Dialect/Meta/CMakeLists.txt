# Copyright (c) 2022-present, Trail of Bits, Inc.
add_mlir_dialect(Meta meta)
add_mlir_doc(Meta Meta Meta/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Meta.td)

mlir_tablegen(MetaAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MetaAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMetaAttributesIncGen)
