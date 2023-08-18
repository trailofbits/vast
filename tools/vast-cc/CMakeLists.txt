get_property(VAST_LIBS GLOBAL PROPERTY VAST_ALL_LIBS)

add_executable(vast-cc
    vast-cc.cpp
)

target_link_libraries(vast-cc
  PRIVATE
    ${VAST_LIBS}
    ${LLVM_LIBS}
    ${CLANG_LIBS}
    ${MLIR_LIBS}
    MLIROptLib

    vast::settings
)

vast_check_link_libraries(vast-cc)
