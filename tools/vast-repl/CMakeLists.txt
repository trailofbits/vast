add_vast_executable(vast-repl
    vast-repl.cpp
    cli.cpp
    codegen.cpp
    command.cpp
    config.cpp
    state.cpp

    LINK_LIBS
      ${CLANG_LIBS}
)
