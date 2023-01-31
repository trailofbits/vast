# Copyright (c) 2023-present, Trail of Bits, Inc.

add_library(vast_codegen
    ABIInfo.cpp
    FunctionInfo.cpp
    Generator.cpp
    Module.cpp
    Passes.cpp
    TargetInfo.cpp
    TypesGenerator.cpp
)

target_link_libraries(vast_codegen PRIVATE vast::settings)

add_library(vast::codegen ALIAS vast_codegen)
