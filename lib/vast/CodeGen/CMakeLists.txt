# Copyright (c) 2023-present, Trail of Bits, Inc.

add_library(vast_codegen
    Generator.cpp
    Module.cpp
    Passes.cpp
)

target_link_libraries(vast_codegen PRIVATE vast::settings)

add_library(vast::codegen ALIAS vast_codegen)
