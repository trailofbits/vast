# Copyright (c) 2023-present, Trail of Bits, Inc.

add_library(vast_codegen
    ABIInfo.cpp
    ArgInfo.cpp
    FunctionInfo.cpp
    Generator.cpp
    Passes.cpp
    TargetInfo.cpp
    TypeInfo.cpp
)

target_link_libraries(vast_codegen PRIVATE vast::settings)

add_library(vast::codegen ALIAS vast_codegen)
