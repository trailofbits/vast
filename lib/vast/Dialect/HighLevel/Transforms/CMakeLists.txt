add_mlir_dialect_library(MLIRHighLevelTransforms
  ExportFnInfo.cpp
  HLLowerTypes.cpp
  HLStructsToLLVM.cpp
  HLToLL.cpp
  HLToSCF.cpp
  LLVMDump.cpp

  DEPENDS
  HighLevelTransformsIncGen

  LINK_LIBS PUBLIC
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
