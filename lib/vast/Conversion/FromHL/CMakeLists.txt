# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_conversion_from_hl
    ToLLCF.cpp
    ToFunc.cpp
    EmitLazyRegions.cpp
    StructsToLLVM.cpp
    ToLLGEPs.cpp
    ToLLVars.cpp
)

target_link_libraries(vast_conversion_from_hl
  PRIVATE
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

add_dependencies(vast_conversion_from_hl VASTConversionPassIncGen)
add_library(vast::conversion::from_hl ALIAS vast_conversion_from_hl)
