// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Dialect/Parser/Dialect.hpp"
#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeSupport.h>

namespace vast::pr {
    using OpBuilder = mlir::OpBuilder;

    void ParserDialect::initialize() {
        registerTypes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/Parser/Parser.cpp.inc"
        >();
    }

    Operation *ParserDialect::materializeConstant(
        OpBuilder &builder, Attribute value, Type type, Location loc
    ) {
        VAST_UNIMPLEMENTED;
    }

} // namespace vast::pr

#include "vast/Dialect/Parser/ParserDialect.cpp.inc"
