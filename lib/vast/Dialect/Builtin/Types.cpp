// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Builtin/Dialect.hpp"
#include "vast/Dialect/Builtin/Types.hpp"

#include "vast/Util/Common.hpp"

namespace vast::hlbi {

    void HLBuiltinDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/Builtin/BuiltinTypes.cpp.inc"
        >();
    }

} // namespace vast::hlbi

#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/Builtin/BuiltinTypes.cpp.inc"
