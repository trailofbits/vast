add_mlir_dialect_library(MLIRLowLevelTransforms
    ToLLVM.cpp

    DEPENDS
    LowLevelTransformsIncGen

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
