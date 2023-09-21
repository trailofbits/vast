add_vast_executable(vast-repl
    vast-repl.cpp
    cli.cpp
    codegen.cpp
    command.cpp
    config.cpp

    LINK_LIBS
      ${CLANG_LIBS}
)
