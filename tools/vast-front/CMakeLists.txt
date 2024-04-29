add_vast_executable(vast-front
  compiler_invocation.cpp
  driver.cpp
  cc1.cpp

  LINK_LIBS
    ${LLVM_LIBS}
    ${CLANG_LIBS}
)

if(VAST_CUSTOM_INCLUDE_PATH)
    add_compile_definitions(VAST_CUSTOM_INCLUDE_PATH="${VAST_CUSTOM_INCLUDE_PATH}")
endif()
