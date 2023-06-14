# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_high_level_transforms STATIC
  ExportFnInfo.cpp
  HLLowerTypes.cpp
  LLVMDump.cpp
  DCE.cpp
  ResolveTypeDefs.cpp
)

target_link_libraries(vast_high_level_transforms
  PUBLIC
    MLIRLowLevel
    MLIRHighLevel
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRExecutionEngine
    MLIRLLVMCommonConversion
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration

    vast::settings
)

add_dependencies(vast_high_level_transforms HighLevelTransformsIncGen)

add_library(vast::high_level::transforms ALIAS vast_high_level_transforms)
