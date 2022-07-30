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

#include "vast/Translation/CodeGenVisitor.hpp"
#include "vast/Translation/CodeGenFallBackVisitor.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Translation/DataLayout.hpp"
#include "vast/Translation/CodeGenMeta.hpp"

namespace vast::hl
{
    namespace detail {
        static inline MContext* codegen_context_setup(MContext *ctx) {
            ctx->loadDialect< hl::HighLevelDialect >();
            ctx->loadDialect< mlir::StandardOpsDialect >();
            ctx->loadDialect< mlir::DLTIDialect >();
            ctx->loadDialect< mlir::scf::SCFDialect >();
            return ctx;
        };

    } // namespace detail

    //
    // CodeGenUnit
    //
    // It takes care of translation of single translation unit or declaration.
    //
    template< typename CodeGenVisitor >
    struct CodeGenBase
    {
        using MetaGenerator = typename CodeGenVisitor::MetaGeneratorType;

        CodeGenBase(MContext *ctx, MetaGenerator &meta)
            : ctx(ctx), meta(meta)
        {
            detail::codegen_context_setup(ctx);
        }

        OwningModuleRef emit_module(clang::ASTUnit *unit) {
            return emit_module_impl(unit);
        }

        OwningModuleRef emit_module(clang::Decl *decl) {
            return emit_module_impl(decl);
        }

    private:
        template< typename AST >
        OwningModuleRef emit_module_impl(AST *ast) {
            // TODO(Heno): fix module location
            OwningModuleRef mod = { Module::create(
                mlir::UnknownLoc::get(ctx)
            ) };

            CodeGenContext cgctx(*ctx, ast->getASTContext(), mod);

            llvm::ScopedHashTableScope type_def_scope(cgctx.type_defs);
            llvm::ScopedHashTableScope type_dec_scope(cgctx.type_decls);
            llvm::ScopedHashTableScope enum_dec_scope(cgctx.enum_decls);
            llvm::ScopedHashTableScope enum_constant_scope(cgctx.enum_constants);
            llvm::ScopedHashTableScope func_scope(cgctx.functions);
            llvm::ScopedHashTableScope glob_scope(cgctx.vars);

            CodeGenVisitor visitor(cgctx, meta);

            process(ast, visitor);

            emit_data_layout(*ctx, mod, cgctx.data_layout());
            // TODO(Heno): verify module
            return mod;
        }

        static bool process_root_decl(void * context, const clang::Decl *decl) {
            CodeGenVisitor &visitor = *static_cast<CodeGenVisitor*>(context);
            return visitor.Visit(decl), true;
        }

        void process(clang::ASTUnit *unit, CodeGenVisitor &visitor) {
            unit->visitLocalTopLevelDecls(&visitor, process_root_decl);
        }

        void process(clang::Decl *decl, CodeGenVisitor &visitor) {
            visitor.Visit(decl);
        }

        MContext *ctx;
        MetaGenerator &meta;
    };

    //
    // DefaultCodeGen
    //
    // Uses `DefaultMetaGenerator` and `DefaultCodeGenVisitorMixin`
    // with `DefaultFallBack` for the generation.
    //
    struct DefaultCodeGen
    {
        template< typename Derived >
        using VisitorMixin = CodeGenFallBackVisitorMixin< Derived,
            DefaultCodeGenVisitorMixin,
            DefaultFallBackVisitorMixin
        >;

        using Visitor = CodeGenVisitor< VisitorMixin >;

        using Base = CodeGenBase< Visitor >;
        using MetaGenerator = Visitor::MetaGeneratorType;

        DefaultCodeGen(MContext *ctx)
            : meta(ctx), codegen(ctx, meta)
        {}

        OwningModuleRef emit_module(clang::ASTUnit *unit) {
            return codegen.emit_module(unit);
        }

        OwningModuleRef emit_module(clang::Decl *decl) {
            return codegen.emit_module(decl);
        }

        MetaGenerator meta;
        CodeGenBase< Visitor > codegen;
    };

} // namespace vast::hl
