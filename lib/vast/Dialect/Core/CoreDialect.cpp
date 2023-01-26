// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>
#include <optional>
#include <type_traits>
#include <vector>

namespace vast::core
{
    void CoreDialect::initialize()
    {
        registerTypes();
        registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/Core/Core.cpp.inc"
        >();
    }

    using OpBuilder = mlir::OpBuilder;

    Operation *CoreDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type, Location loc)
    {
        VAST_UNIMPLEMENTED;
    }
} // namespace vast::core

#include "vast/Dialect/Core/CoreDialect.cpp.inc"
