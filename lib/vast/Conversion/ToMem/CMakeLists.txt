# Copyright (c) 2024-present, Trail of Bits, Inc.

add_vast_conversion_library(ToMemConversionPasses
    RefsToSSA.cpp
    StripParamLValues.cpp
    VarsToCells.cpp
)
