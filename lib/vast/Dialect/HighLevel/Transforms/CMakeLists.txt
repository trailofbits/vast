add_mlir_dialect_library(MLIRHighLevelTransforms
  LowerTypes.cpp
  LowerControlFlow.cpp

  DEPENDS
  HighLevelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRHighLevel
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
