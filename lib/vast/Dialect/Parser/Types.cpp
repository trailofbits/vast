// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Parser/Dialect.hpp"
#include "vast/Dialect/Parser/Types.hpp"

#include "vast/Util/Common.hpp"

namespace vast::pr {

    void ParserDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/Parser/ParserTypes.cpp.inc"
        >();
    }

} // namespace vast::pr

#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/Parser/ParserTypes.cpp.inc"
