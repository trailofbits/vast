# Copyright (c) 2022-present, Trail of Bits, Inc.
add_mlir_dialect(ABI abi)
add_mlir_doc(ABI ABI ABI/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ABI.td)
