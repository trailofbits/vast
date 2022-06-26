// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Meta/MetaTypes.hpp"

using StringRef = llvm::StringRef; // to fix missing namespace in generated file

namespace vast::meta {
    void MetaDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/Meta/MetaTypes.cpp.inc"
        >();
    }
} // namespace vast::meta

#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/Meta/MetaTypes.cpp.inc"
