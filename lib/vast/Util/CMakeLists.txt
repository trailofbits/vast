# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_util
    Region.cpp
    Warnings.cpp
)

target_link_libraries(vast_util PRIVATE vast::settings)

add_library(vast::util ALIAS vast_util)
