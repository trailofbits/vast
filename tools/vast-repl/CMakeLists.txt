get_property(VAST_DIALECT_LIBS GLOBAL PROPERTY VAST_DIALECT_LIBS)

add_executable(vast-repl
    vast-repl.cpp
    codegen.cpp
    command.cpp
)

target_link_libraries(vast-repl
  PRIVATE
    ${VAST_DIALECT_LIBS}
    ${MLIR_DIALECT_LIBS}

    VASTUtil
    VASTSymbolInterface
    VASTCodeGen

    ${CLANG_LIBS}

    vast::settings
)

vast_check_link_libraries(vast-repl)
