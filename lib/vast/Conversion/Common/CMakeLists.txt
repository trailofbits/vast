# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_conversion_common
    IRsToLLVM.cpp
)

target_link_libraries(vast_conversion_common
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

add_dependencies(vast_conversion_common VASTConversionPassIncGen)
add_library(vast::conversion::common ALIAS vast_conversion_common)
