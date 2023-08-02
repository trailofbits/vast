# Copyright (c) 2021-present, Trail of Bits, Inc.
add_mlir_dialect(HighLevel hl)
add_mlir_doc(HighLevel HighLevel HighLevel/ -gen-dialect-doc -dialect=hl)

set(LLVM_TARGET_DEFINITIONS HighLevel.td)
mlir_tablegen(HighLevelEnums.h.inc -gen-enum-decls)
mlir_tablegen(HighLevelEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHighLevelEnumsIncGen)

mlir_tablegen(HighLevelAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(HighLevelAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRHighLevelAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(HighLevelTransformsIncGen)
add_mlir_doc(Passes  HighLevelPasses ./ -gen-pass-doc)
