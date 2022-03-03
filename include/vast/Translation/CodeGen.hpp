// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Expr.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

namespace vast::hl
{
    using module_owning_ref = mlir::OwningModuleRef;
    using module_t          = mlir::ModuleOp;
    using context_t         = mlir::MLIRContext;

    struct high_level_codegen {
        high_level_codegen(context_t *ctx)
            : ctx(ctx)
        {
            ctx->loadDialect< HighLevelDialect >();
            ctx->loadDialect< mlir::StandardOpsDialect >();
            ctx->loadDialect< mlir::DLTIDialect >();
        }

        module_owning_ref emit_module(clang::Decl *decl);

      private:
        context_t *ctx;
    };

} // namespace vast::hl
