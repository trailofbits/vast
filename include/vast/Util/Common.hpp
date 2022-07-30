// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Expr.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
VAST_UNRELAX_WARNINGS

#include <variant>

namespace vast
{
    using Value       = mlir::Value;
    using Operation   = mlir::Operation;

    using AContext = clang::ASTContext;
    using MContext = mlir::MLIRContext;

    using Module          = mlir::ModuleOp;
    using OwningModuleRef = mlir::OwningOpRef< Module >;
} // namespace vast
