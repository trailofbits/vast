# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_dialect_library(VASTMeta
    MetaAttributes.cpp
    MetaDialect.cpp
    MetaTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast

    DEPENDS
    VASTMetaAttributesIncGen

    LINK_LIBS PUBLIC

    MLIRIR
    MLIRSupport
)

target_link_libraries(VASTMeta PRIVATE vast::settings)
