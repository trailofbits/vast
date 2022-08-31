# Copyright (c) 2022-present, Trail of Bits, Inc.
add_mlir_dialect(LowLevel ll)
add_mlir_doc(LowLevel LowLevel LowLevel/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS LowLevel.td)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(LowLevelTransformsIncGen)
add_mlir_doc(Passes LowLevelPasses ./ -gen-pass-doc)
