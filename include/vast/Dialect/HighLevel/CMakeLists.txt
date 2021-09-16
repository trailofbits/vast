# Copyright (c) 2021-present, Trail of Bits, Inc.
add_mlir_dialect(HighLevel hl)
add_mlir_doc(HighLevel -gen-dialect-doc HighLevel Dialects/)

set(LLVM_TARGET_DEFINITIONS HighLevel.td)
mlir_tablegen(HighLevelEnums.h.inc -gen-enum-decls)
mlir_tablegen(HighLevelEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHighLevelEnumsIncGen)

# set(LLVM_TARGET_DEFINITIONS HighLevelAttributes.td)
# mlir_tablegen(HighLevelAttributes.h.inc -gen-attrdef-decls)
# mlir_tablegen(HighLevelAttributes.cpp.inc -gen-attrdef-defs)
# add_public_tablegen_target(MLIRHighLevelAttributesIncGen)
