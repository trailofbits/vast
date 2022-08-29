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
#include <mlir/InitAllDialects.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include "vast/Translation/CodeGenVisitor.hpp"
#include "vast/Translation/CodeGenFallBackVisitor.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/Meta/MetaDialect.hpp"
#include "vast/Dialect/Dialects.hpp"

#include "vast/Translation/DataLayout.hpp"
#include "vast/Translation/CodeGenMeta.hpp"

namespace vast::hl
{
    namespace detail {
        static inline MContext* codegen_context_setup(MContext *ctx) {
            ctx->loadDialect< hl::HighLevelDialect >();
            ctx->loadDialect< meta::MetaDialect >();
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

        CodeGenBase(MContext *mctx, MetaGenerator &meta)
            : _mctx(mctx), _meta(meta), _cgctx(nullptr), _module(nullptr)
        {
            detail::codegen_context_setup(_mctx);
        }

        OwningModuleRef emit_module(clang::ASTUnit *unit) {
            append_to_module(unit);
            return freeze();
        }

        OwningModuleRef emit_module(clang::Decl *decl) {
            append_to_module(decl);
            return freeze();
        }

        void append_to_module(clang::ASTUnit *unit) { append_impl(unit); }

        void append_to_module(clang::Decl *decl) { append_impl(decl); }

        void append_to_module(clang::Stmt *stmt) { append_impl(stmt); }

        void append_to_module(clang::Expr *expr) { append_impl(expr); }

        void append_to_module(clang::Type *type) { append_impl(type); }

        OwningModuleRef freeze() {
            emit_data_layout(*_mctx, _module, _cgctx->data_layout());
            return std::move(_module);
        }

        template< typename From, typename Symbol >
        using ScopedSymbolTable = llvm::ScopedHashTableScope< From, Symbol >;

        using TypeDefsScope      = ScopedSymbolTable< const clang::TypedefDecl*, TypeDefOp >;
        using TypeDeclsScope     = ScopedSymbolTable< StringRef, TypeDeclOp >;
        using EnumDeclsScope     = ScopedSymbolTable< StringRef, EnumDeclOp >;
        using EnumConstantsScope = ScopedSymbolTable< StringRef, EnumConstantOp >;
        using FunctionsScope     = ScopedSymbolTable< StringRef, mlir::FuncOp >;
        using VariablesScope     = ScopedSymbolTable< const clang::VarDecl*, Value >;

        struct CodegenScope {
            TypeDefsScope      typedefs;
            TypeDeclsScope     typedecls;
            EnumDeclsScope     enumdecls;
            EnumConstantsScope enumconsts;
            FunctionsScope     funcs;
            VariablesScope     globs;
        };

    private:

        void setup_codegen(AContext &actx) {
            if (_cgctx)
                return;

            // TODO(Heno): fix module location
            _module = { Module::create(mlir::UnknownLoc::get(_mctx)) };

            _cgctx = std::make_unique< CodeGenContext >(*_mctx, actx, _module);

            _scope = std::unique_ptr< CodegenScope >( new CodegenScope{
                _cgctx->typedefs,
                _cgctx->type_decls,
                _cgctx->enum_decls,
                _cgctx->enum_constants,
                _cgctx->functions,
                _cgctx->vars
            });

            _visitor = std::make_unique< CodeGenVisitor >(*_cgctx, _meta);
        }

        template< typename AST >
        void append_impl(AST ast) {
            setup_codegen(ast->getASTContext());
            process(ast, *_visitor);
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

        MContext *_mctx;
        MetaGenerator &_meta;

        std::unique_ptr< CodeGenContext > _cgctx;
        std::unique_ptr< CodegenScope >   _scope;
        std::unique_ptr< CodeGenVisitor > _visitor;

        OwningModuleRef _module;
    };

    template< typename Derived >
    using DefaultCodeGenVisitorConfig = CodeGenFallBackVisitorMixin< Derived,
        DefaultCodeGenVisitorMixin,
        DefaultFallBackVisitorMixin
    >;

    //
    // DefaultCodeGen
    //
    // Uses `DefaultMetaGenerator` and `DefaultCodeGenVisitorMixin`
    // with `DefaultFallBack` for the generation.
    //
    template<
        template< typename >
        typename VisitorConfig = DefaultCodeGenVisitorConfig,
        typename MetaGenerator = DefaultMetaGenerator
    >
    struct DefaultCodeGen
    {
        using Visitor = CodeGenVisitor< VisitorConfig, MetaGenerator >;

        using Base = CodeGenBase< Visitor >;

        DefaultCodeGen(AContext *actx, MContext *mctx)
            : meta(actx, mctx), codegen(mctx, meta)
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

    using CodeGenWithMetaIDs = DefaultCodeGen< DefaultCodeGenVisitorConfig, IDMetaGenerator >;

} // namespace vast::hl
