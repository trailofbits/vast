get_property(VAST_LIBS GLOBAL PROPERTY VAST_ALL_LIBS)

add_executable(vast-front
  compiler_invocation.cpp
  driver.cpp
  cc1.cpp
)

target_link_libraries(vast-front
  PRIVATE
    ${VAST_LIBS}
    ${LLVM_LIBS}
    ${CLANG_LIBS}
    ${MLIR_LIBS}

    vast::settings
)

vast_check_link_libraries(vast-front)
