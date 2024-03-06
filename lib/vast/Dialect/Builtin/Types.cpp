// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Builtin/Dialect.hpp"
#include "vast/Dialect/Builtin/Ops.hpp"
#include "vast/Dialect/Builtin/Types.hpp"

namespace vast::builtin {
    void BuiltinDialect::registerTypes() {
        addTypes<
#define GET_TYPEDEF_LIST
#include "vast/Dialect/Builtin/BuiltinTypes.cpp.inc"
            >();
    }
} // namespace vast::builtin

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/Builtin/BuiltinTypes.cpp.inc"
