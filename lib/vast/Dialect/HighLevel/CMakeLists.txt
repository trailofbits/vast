# Copyright (c) 2021-present, Trail of Bits, Inc.

add_vast_dialect_library(HighLevel
    HighLevelDialect.cpp
    HighLevelVar.cpp
    HighLevelOps.cpp
    HighLevelAttributes.cpp
    HighLevelTypes.cpp
    Passes.cpp

    LINK_LIBS PRIVATE
        VASTAliasTypeInterface
        VASTElementTypeInterface
)

add_subdirectory(Transforms)
