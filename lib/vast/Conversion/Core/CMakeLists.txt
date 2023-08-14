# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_conversion_library(VASTCoreConversionPasses
    ToLLVM.cpp

    DEPENDS
        VASTConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRLowLevel
        VASTHighLevel
        MLIRIR
        MLIRPass
        MLIRTransformUtils
        MLIRExecutionEngine
        MLIRLLVMCommonConversion
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        MLIRToLLVMIRTranslationRegistration
)

add_library(vast_conversion_core
    ToLLVM.cpp
)

target_link_libraries(vast_conversion_core
  PRIVATE
    MLIRLowLevel
    VASTHighLevel
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

add_dependencies(vast_conversion_core VASTConversionPassIncGen)
add_library(vast::conversion::core ALIAS vast_conversion_core)
