add_vast_executable(vast-front
  compiler_invocation.cpp
  driver.cpp
  cc1.cpp

  LINK_LIBS
    ${LLVM_LIBS}
    ${CLANG_LIBS}
)
