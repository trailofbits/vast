
set(VAST_TARGET_DEFINITIONS Passes.td)
vast_tablegen(Passes.h.inc -gen-pass-decls -name ParserConversion)
vast_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ParserConversion)
vast_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ParserConversion)
add_public_vast_tablegen_target(VASTParserConversionPassIncGen)
add_dependencies(vast-headers VASTParserConversionPassIncGen)
add_mlir_doc(ParserPasses ParserConversionPasses ./ -gen-pass-doc)

set(PARSER_CONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/Config.hpp
)

file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/default-parsers-config.yaml
    DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR}
)
