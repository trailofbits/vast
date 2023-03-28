# Copyright (c) 2022-present, Trail of Bits, Inc.
add_mlir_dialect_library(MLIRCore
    CoreDialect.cpp
    CoreOps.cpp
    CoreTypes.cpp
    CoreTraits.cpp
    CoreAttributes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    DEPENDS
    MLIRCoreIncGen
    MLIRCoreAttributesIncGen

    LINK_LIBS PUBLIC

    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces

    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRControlFlowInterfaces
    MLIRDataLayoutInterfaces
    MLIRInferTypeOpInterface

    VASTSymbolInterface
    VASTTypeQualifiersInterfaces
)

target_link_libraries(MLIRCore PRIVATE vast_settings)

