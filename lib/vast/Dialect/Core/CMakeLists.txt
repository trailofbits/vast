# Copyright (c) 2022-present, Trail of Bits, Inc.
add_vast_dialect_library(Core
    CoreAttributes.cpp
    CoreDialect.cpp
    CoreOps.cpp
    CoreTraits.cpp
    CoreTypes.cpp
    Func.cpp
    Linkage.cpp
    SymbolTable.cpp

    LINK_LIBS PRIVATE
        VASTAliasTypeInterface
)
