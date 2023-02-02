// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Expr.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OwningOpRef.h>
VAST_UNRELAX_WARNINGS

#include <variant>
#include <optional>

namespace vast
{
    using Type = mlir::Type;
    using Value = mlir::Value;
    using Attribute = mlir::Attribute;
    using Operation = mlir::Operation;
    using Block = mlir::Block;

    using Region    = mlir::Region;
    using Builder   = mlir::OpBuilder;
    using Location  = mlir::Location;
    using State     = mlir::OperationState;
    using TypeRange = mlir::TypeRange;

    using InsertionGuard = Builder::InsertionGuard;

    using Parser      = mlir::OpAsmParser;
    using ParseResult = mlir::ParseResult;
    using Printer     = mlir::OpAsmPrinter;
    using FoldResult  = mlir::OpFoldResult;

    using BuilderCallback = std::optional<
        llvm::function_ref< void(Builder &, Location) >
    >;

    using acontext_t = clang::ASTContext;
    using mcontext_t = mlir::MLIRContext;

    // FIXME: eventually replace with tower_module
    using vast_module = mlir::ModuleOp;
    using owning_module_ref = mlir::OwningOpRef< vast_module >;

    using mlir_type  = mlir::Type;
    using clang_type = clang::Type;

    using string_ref     = llvm::StringRef;
    using logical_result = mlir::LogicalResult;

} // namespace vast
