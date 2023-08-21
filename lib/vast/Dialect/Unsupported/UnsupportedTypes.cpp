// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"

namespace vast::unsup {
    void UnsupportedDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/Unsupported/UnsupportedTypes.cpp.inc"
        >();
    }
} // namespace vast::unsup

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/Unsupported/UnsupportedTypes.cpp.inc"
