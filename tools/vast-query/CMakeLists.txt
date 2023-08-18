get_property(VAST_DIALECT_LIBS GLOBAL PROPERTY VAST_DIALECT_LIBS)

add_executable(vast-query
    vast-query.cpp
)

target_link_libraries(vast-query
  PRIVATE
    ${VAST_DIALECT_LIBS}
    ${MLIR_DIALECT_LIBS}

    VASTUtil
    VASTSymbolInterface
    vast::settings
)

vast_check_link_libraries(vast-query)
