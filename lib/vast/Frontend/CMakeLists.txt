# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_frontend
    GenAction.cpp
    Options.cpp
)

mk_clang_libs(frontend_clang_libs clangCodeGen)
target_link_libraries(vast_frontend
    PUBLIC
        vast::target_llvmir

        ${frontend_clang_libs}
    PRIVATE
        vast::settings)

add_library(vast::frontend ALIAS vast_frontend)

# TODO: Move to vast_add_library
set_property(GLOBAL APPEND PROPERTY VAST_INSTALL_TARGETS vast_frontend)
