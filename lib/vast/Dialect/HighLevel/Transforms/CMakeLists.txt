add_mlir_dialect_library(MLIRHighLevelTransforms
  ExportFnInfo.cpp
  HLLowerTypes.cpp
  HLStructsToLLVM.cpp
  LLVMDump.cpp
  HLToLLGEPs.cpp
  HLToLLVars.cpp
  HLEmitLazyRegions.cpp
  DCE.cpp

  DEPENDS
  HighLevelTransformsIncGen

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
