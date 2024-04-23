# Copyright (c) 2022-present, Trail of Bits, Inc.

add_vast_conversion_library(HighLevelTransforms
  ExportFnInfo.cpp
  HLLowerTypes.cpp
  DCE.cpp
  LowerElaboratedTypes.cpp
  LowerEnums.cpp
  LowerTypeDefs.cpp
  SpliceTrailingScopes.cpp
  UDE.cpp
)
