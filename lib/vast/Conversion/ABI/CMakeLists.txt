# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_conversion_abi STATIC
    EmitABI.cpp
    LowerABI.cpp
)

target_link_libraries(vast_conversion_abi
  PRIVATE
    MLIRLowLevel
    MLIRHighLevel
    MLIRIR
    MLIRPass
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

add_dependencies(vast_conversion_abi VASTConversionPassIncGen)

add_library(vast::conversion::abi ALIAS vast_conversion_abi)
