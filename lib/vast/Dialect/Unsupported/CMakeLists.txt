# Copyright (c) 2023-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRUnsupported
    UnsupportedDialect.cpp
    UnsupportedOps.cpp
    UnsupportedTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vast
    
    DEPENDS
    MLIRUnsupportedIncGen

    LINK_LIBS PUBLIC

    MLIRIR
    MLIRSupport
)

target_link_libraries(MLIRUnsupported PRIVATE vast::settings)