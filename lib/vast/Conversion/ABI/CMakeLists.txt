# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_conversion_library(VASTABIConversionPasses
    EmitABI.cpp

    DEPENDS
        VASTConversionPassIncGen

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
