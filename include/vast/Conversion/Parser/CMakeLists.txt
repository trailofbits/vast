
set(VAST_TARGET_DEFINITIONS Passes.td)
vast_tablegen(Passes.h.inc -gen-pass-decls -name ParserConversion)
vast_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ParserConversion)
vast_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ParserConversion)
add_public_vast_tablegen_target(VASTParserConversionPassIncGen)
add_dependencies(vast-headers VASTParserConversionPassIncGen)
add_mlir_doc(ParserPasses ParserConversionPasses ./ -gen-pass-doc)
