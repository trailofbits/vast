# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRMeta
    MetaDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    LINK_LIBS PUBLIC
        MLIRIR
)

target_link_libraries(MLIRHighLevel PRIVATE vast_settings)
