# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRMeta
    MetaAttributes.cpp
    MetaDialect.cpp
    MetaTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    DEPENDS
    MLIRMetaAttributesIncGen

    LINK_LIBS PUBLIC
        MLIRIR
)

target_link_libraries(MLIRMeta PRIVATE vast_settings)
