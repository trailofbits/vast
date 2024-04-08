# Copyright (c) 2022-present, Trail of Bits, Inc.



set(VAST_TARGET_DEFINITIONS Passes.td)
vast_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
vast_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Conversion)
vast_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Conversion)
add_public_vast_tablegen_target(VASTConversionPassIncGen)
add_dependencies(vast-headers VASTConversionPassIncGen)
add_mlir_doc(Passes ConversionPasses ./ -gen-pass-doc)
