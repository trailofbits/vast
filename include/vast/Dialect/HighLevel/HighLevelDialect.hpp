// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
VAST_RELAX_WARNINGS

#include <optional>

namespace vast::hl
{
    using Type = mlir::Type;
    using Value = mlir::Value;
    using Attribute = mlir::Attribute;
    using Operation = mlir::Operation;

    using Region    = mlir::Region;
    using Builder   = mlir::OpBuilder;
    using Location  = mlir::Location;
    using State     = mlir::OperationState;
    using TypeRange = mlir::TypeRange;

    using Parser      = mlir::OpAsmParser;
    using ParseResult = mlir::ParseResult;

    using Printer     = mlir::OpAsmPrinter;

    using LogicalResult = mlir::LogicalResult;

    using FoldResult = mlir::OpFoldResult;

    using BuilderCallback = std::optional<
        llvm::function_ref< void(Builder &, Location) >
    >;

} // namespace vast::hl

// Pull in the dialect definition.
#include "vast/Dialect/HighLevel/HighLevelDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "vast/Dialect/HighLevel/HighLevelEnums.h.inc"


