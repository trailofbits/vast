# Copyright (c) 2022-present, Trail of Bits, Inc.

add_vast_conversion_library(HighLevelConversionPasses
    ToLLCF.cpp
    EmitLazyRegions.cpp
    StructsToLLVM.cpp
    ToLLGEPs.cpp
    ToLLVars.cpp
    ToLLFunc.cpp
)
