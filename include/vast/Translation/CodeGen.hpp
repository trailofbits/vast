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

#include "vast/Util/Common.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

namespace vast::hl
{
    struct high_level_codegen {
        high_level_codegen(MContext *ctx)
            : ctx(ctx)
        {
            ctx->loadDialect< HighLevelDialect >();
            ctx->loadDialect< mlir::StandardOpsDialect >();
            ctx->loadDialect< mlir::DLTIDialect >();
        }

        OwningModuleRef emit_module(clang::Decl *decl);

      private:
        MContext *ctx;
    };

} // namespace vast::hl
