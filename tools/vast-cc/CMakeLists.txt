add_vast_executable(vast-cc
    vast-cc.cpp

    LINK_LIBS
      ${LLVM_LIBS}
      ${CLANG_LIBS}
      MLIROptLib
)
