add_vast_executable(vast-repl
    vast-repl.cpp
    codegen.cpp
    command.cpp

    LINK_LIBS
      ${CLANG_LIBS}
)
