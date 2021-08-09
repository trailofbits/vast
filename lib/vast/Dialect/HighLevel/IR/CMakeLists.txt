# Copyright (c) 2021-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRHighLevel
    HighLevel.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    DEPENDS
    MLIRHighLevelIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSideEffectInterfaces
)

target_link_libraries(MLIRHighLevel PRIVATE vast_settings)
