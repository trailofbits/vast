get_property(VAST_LIBS GLOBAL PROPERTY VAST_ALL_LIBS)

add_executable(vast-opt
    vast-opt.cpp
)

target_link_libraries(vast-opt
  PRIVATE
    ${VAST_LIBS}
    ${MLIR_LIBS}
    MLIROptLib

    vast::settings
)

vast_check_link_libraries(vast-opt)
