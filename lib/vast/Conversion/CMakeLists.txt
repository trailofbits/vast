# Copyright (c) 2022-present, Trail of Bits, Inc.

if (ENABLE_PDLL_CONVERSIONS)

    add_mlir_conversion_library( VASTHLToFunc
        HLToFunc.cpp

        ADDITIONAL_HEADER_DIRS
            ${PROJECT_SOURCE_DIR}/include/vast

        DEPENDS
            VASTHLToFuncIncGen
            VASTConversionPassIncGen

        LINK_COMPONENTS
            Core

        LINK_LIBS PUBLIC
            MLIRIR
            MLIRPass
            MLIRSupport
            MLIRTransformUtils

            MLIRMeta
            MLIRHighLevel

            vast_settings
    )

endif() # ENABLE_PDLL_CONVERSIONS

add_library( VASTConversions INTERFACE )

add_dependencies( VASTConversions
    VASTConversionPassIncGen
)

add_mlir_dialect_library(VastConversionPasses
    ToLLVM.cpp

    DEPENDS
        VASTConversionPassesIncGen

    LINK_LIBS PUBLIC
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
)
