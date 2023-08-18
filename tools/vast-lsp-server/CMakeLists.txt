get_property(VAST_LIBS GLOBAL PROPERTY VAST_ALL_LIBS)

add_executable(vast-lsp-server
    vast-lsp-server.cpp
)

target_link_libraries(vast-lsp-server
  PRIVATE
    ${VAST_LIBS}
    ${MLIR_LIBS}
    MLIRLspServerLib
    vast::settings
)
