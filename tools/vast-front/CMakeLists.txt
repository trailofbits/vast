add_vast_executable(vast-front
  compiler_invocation.cpp
  driver.cpp
  cc1.cpp

  LINK_LIBS
    ${LLVM_LIBS}
    ${CLANG_LIBS}
)

set(CLANG_BIN_PATH ${LLVM_BUILD_BINARY_DIR}/bin/clang)

if(NOT EXISTS ${CLANG_BIN_PATH})
    message(FATAL_ERROR "Could not find clang binary to deduce resource path!")
endif()

add_compile_definitions(CLANG_BINARY_PATH="${CLANG_BIN_PATH}")
