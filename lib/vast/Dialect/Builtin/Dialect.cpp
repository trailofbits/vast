// Copyright (c) 2024-present, Trail of Bits, Inc.
//
#include "vast/Dialect/Builtin/Dialect.hpp"
#include "vast/Dialect/Builtin/Ops.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeSupport.h>

namespace vast::hlbi {
    using OpBuilder = mlir::OpBuilder;

    void HLBuiltinDialect::initialize() {
        addOperations<
#define GET_OP_LIST
#include "vast/Dialect/Builtin/Builtin.cpp.inc"
            >();
    }

    Operation *HLBuiltinDialect::materializeConstant(
        OpBuilder &builder, Attribute value, Type type, Location loc
    ) {
        VAST_UNIMPLEMENTED;
    }
} // namespace vast::hlbi

#include "vast/Dialect/Builtin/BuiltinDialect.cpp.inc"
