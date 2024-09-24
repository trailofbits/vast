# Copyright (c) 2024-present, Trail of Bits, Inc.

add_vast_conversion_library(ToMemConversionPasses
    EvictStaticLocals.cpp
    RefsToSSA.cpp
    StripParamLValues.cpp
    VarsToCells.cpp
)
