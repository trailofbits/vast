// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"
#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
VAST_RELAX_WARNINGS

namespace vast::us {
    mlir::Type strip_unsupported(mlir::Value v) { return strip_unsupported(v.getType()); }

    mlir::Type strip_unsupported(mlir::Type t) {
        if (auto e = mlir::dyn_cast< us::UnsupportedType >(t)) {
            return e.getElementType();
        }
        return t;
    }

    void UnsupportedDialect::registerTypes() {
        addTypes<
#define GET_TYPEDEF_LIST
#include "vast/Dialect/Unsupported/UnsupportedTypes.cpp.inc"
            >();
    }
} // namespace vast::us

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/Unsupported/UnsupportedTypes.cpp.inc"
