# Copyright (c) 2022-present, Trail of Bits, Inc.

add_vast_conversion_library(HighLevelTransforms
  HLLowerTypes.cpp
  DCE.cpp
  LowerElaboratedTypes.cpp
  LowerTypeDefs.cpp
  SpliceTrailingScopes.cpp
  UDE.cpp

  Passes.cpp
)
