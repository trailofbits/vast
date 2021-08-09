
# Copyright (c) 2021-present, Trail of Bits, Inc.

add_subdirectory(HighLevel)

add_mlir_dialect_library(MLIRVast
    VastDialect.cpp
    VastOps.cpp
    VastTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    DEPENDS
    MLIRVastOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)

target_link_libraries(MLIRVast PRIVATE vast_settings)
