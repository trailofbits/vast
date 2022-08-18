# Copyright (c) 2021-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRHighLevel
    HighLevelDialect.cpp
    HighLevelVar.cpp
    HighLevelOps.cpp
    HighLevelAttributes.cpp
    HighLevelTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    DEPENDS
    MLIRHighLevelIncGen
    MLIRHighLevelAttributesIncGen
    MLIRHighLevelEnumsIncGen

    LINK_LIBS PUBLIC

    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces

    MLIRCallInterfaces
    MLIRControlFlowInterfaces
    MLIRDataLayoutInterfaces
    MLIRInferTypeOpInterface

    VASTSymbolInterface
)

target_link_libraries(MLIRHighLevel PRIVATE vast_settings)

add_subdirectory(Transforms)
