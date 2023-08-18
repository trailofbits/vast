# Copyright (c) 2022-present, Trail of Bits, Inc.

add_vast_dialect_library(LowLevel
    LowLevelDialect.cpp
    LowLevelOps.cpp
)

add_subdirectory(Transforms)
