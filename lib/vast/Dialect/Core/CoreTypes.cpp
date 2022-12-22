// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

using StringRef = llvm::StringRef; // to fix missing namespace in generated file

namespace vast::core {
    void CoreDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/Core/CoreTypes.cpp.inc"
        >();
    }
} // namespace vast::core

#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/Core/CoreTypes.cpp.inc"
