set(LLVM_LINK_COMPONENTS Core Support AsmParser)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(VAST_LSP_SERVER_LIBS
  ${DIALECT_LIBS}
  ${CONVERSION_LIBS}

  MLIRHighLevel
  MLIRHighLevelTransforms

  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  vast_settings
)

add_llvm_tool(vast-lsp-server
    vast-lsp-server.cpp

  DEPENDS
    ${VAST_LSP_SERVER_LIBS}
)

target_link_libraries(vast-lsp-server PRIVATE ${VAST_LSP_SERVER_LIBS})
llvm_update_compile_flags(vast-lsp-server)

mlir_check_all_link_libraries(vast-lsp-server)
