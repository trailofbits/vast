// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Expr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
VAST_UNRELAX_WARNINGS

#include <optional>
#include <variant>

namespace vast {
    using Type      = mlir::Type;
    using Value     = mlir::Value;
    using Attribute = mlir::Attribute;
    using Operation = mlir::Operation;
    using Block     = mlir::Block;

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

    using BuilderCallback = std::optional< llvm::function_ref< void(Builder &, Location) > >;

    using BuilderCallBackFn = std::function< void(Builder &, Location) >;

    using acontext_t = clang::ASTContext;
    using mcontext_t = mlir::MLIRContext;

    // FIXME: eventually replace with tower_module
    using vast_module       = mlir::ModuleOp;
    using owning_module_ref = mlir::OwningOpRef< vast_module >;

    using mlir_type  = mlir::Type;

    using operation  = mlir::Operation *;
    using mlir_value = mlir::Value;
    using op_operand = mlir::OpOperand;

    using ap_int         = llvm::APInt;
    using ap_sint        = llvm::APSInt;
    using ap_float       = llvm::APFloat;
    using string_ref     = llvm::StringRef;
    using logical_result = mlir::LogicalResult;

    using insertion_guard = Builder::InsertionGuard;

    using mlir_attr    = mlir::Attribute;

    // TODO replace with attr_t
    using attr_t       = mlir::Attribute;
    using maybe_attr_t = std::optional< mlir::Attribute >;

    using attrs_t       = mlir::SmallVector< mlir::Attribute >;
    using maybe_attrs_t = std::optional< attrs_t >;

    using types_t       = mlir::SmallVector< mlir_type >;
    using maybe_type_t  = std::optional< mlir_type >;
    using maybe_types_t = std::optional< types_t >;

    using values_t      = mlir::SmallVector< mlir_value >;

    using loc_t         = mlir::Location;

    using mlir_builder  = mlir::OpBuilder;

    using region_t      = mlir::Region;
    using region_ptr    = region_t*;

    using block_t       = mlir::Block;
    using block_ptr     = block_t*;

    using walk_result = mlir::WalkResult;

    //
    // clang related aliases
    //
    using clang_function = clang::FunctionDecl;
} // namespace vast
