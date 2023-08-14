# Copyright (c) 2021-present, Trail of Bits, Inc.
add_vast_dialect_with_doc(HighLevel hl)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(HighLevelTransformsIncGen)
add_mlir_doc(Passes  HighLevelPasses ./ -gen-pass-doc)
