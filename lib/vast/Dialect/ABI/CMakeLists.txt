# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRABI
    ABIDialect.cpp
    ABIOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    DEPENDS
    MLIRABIIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    VASTSymbolInterface
)

target_link_libraries(MLIRABI PRIVATE vast_settings)
