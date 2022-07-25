// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Expr.h>
#include <clang/Frontend/ASTUnit.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include "vast/Translation/HighLevelVisitor.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Translation/DataLayout.hpp"

namespace vast::hl
{
    struct high_level_codegen {
        high_level_codegen(MContext *ctx)
            : ctx(ctx)
        {
            ctx->loadDialect< HighLevelDialect >();
            ctx->loadDialect< mlir::StandardOpsDialect >();
            ctx->loadDialect< mlir::DLTIDialect >();
            ctx->loadDialect< mlir::scf::SCFDialect >();
        }

        OwningModuleRef emit_module(clang::ASTUnit *unit, const CodeGenVisitorConfig &config);
        OwningModuleRef emit_module(clang::Decl *decl, const CodeGenVisitorConfig &config);

      private:

        template< typename AST >
        OwningModuleRef process_ast(AST *ast, const CodeGenVisitorConfig &config) {
            mlir::Builder bld(ctx);
            OwningModuleRef mod = {Module::create(bld.getUnknownLoc())};
            TranslationContext tctx(*ctx, ast->getASTContext(), mod);

            llvm::ScopedHashTableScope type_def_scope(tctx.type_defs);
            llvm::ScopedHashTableScope type_dec_scope(tctx.type_decls);
            llvm::ScopedHashTableScope enum_dec_scope(tctx.enum_decls);
            llvm::ScopedHashTableScope enum_constant_scope(tctx.enum_constants);
            llvm::ScopedHashTableScope func_scope(tctx.functions);
            llvm::ScopedHashTableScope glob_scope(tctx.vars);

            CodeGenVisitor visitor(tctx, config);
            process(ast, visitor);

            emit_data_layout(*ctx, mod, tctx.data_layout());
            // TODO(Heno): verify module
            return mod;
        }

        void process(clang::ASTUnit *unit, CodeGenVisitor &visitor);
        void process(clang::Decl *decl, CodeGenVisitor &visitor);

        MContext *ctx;
    };

} // namespace vast::hl
