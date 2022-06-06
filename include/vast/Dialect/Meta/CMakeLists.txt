# Copyright (c) 2022-present, Trail of Bits, Inc.
add_mlir_dialect(Meta meta)
add_mlir_doc(Meta Meta Meta/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Meta.td)
