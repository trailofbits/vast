// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Expr.h>
#include <clang/Frontend/ASTUnit.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Verifier.h>
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
#include "vast/Translation/Types.hpp"

namespace vast::cg
{
    namespace detail {
        static inline mcontext_t& codegen_context_setup(mcontext_t &ctx) {
            mlir::registerAllDialects(ctx);
            vast::registerAllDialects(ctx);

            ctx.loadAllAvailableDialects();
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

        CodeGenBase(mcontext_t *mctx, MetaGenerator &meta)
            : _mctx(mctx), _meta(meta), _cgctx(nullptr), _module(nullptr)
        {
            detail::codegen_context_setup(*_mctx);
        }

        CodeGenBase(acontext_t *actx, mcontext_t *mctx, MetaGenerator &meta)
            : CodeGenBase(mctx, meta)
        {
            setup_codegen(*actx);
        }


        owning_module_ref emit_module(clang::ASTUnit *unit) {
            append_to_module(unit);
            return freeze();
        }

        owning_module_ref emit_module(clang::Decl *decl) {
            append_to_module(decl);
            return freeze();
        }

        void append_to_module(clang::ASTUnit *unit) { append_impl(unit); }

        void append_to_module(clang::Decl *decl) { append_impl(decl); }

        void append_to_module(clang::Stmt *stmt) { append_impl(stmt); }

        void append_to_module(clang::Expr *expr) { append_impl(expr); }

        void append_to_module(clang::Type *type) { append_impl(type); }

        owning_module_ref freeze() {
            hl::emit_data_layout(*_mctx, _module, _cgctx->data_layout());
            return std::move(_module);
        }

        operation build_function_prototype(clang::GlobalDecl decl, mlir_type fty) {
            return _visitor->build_function_prototype(decl, fty);
        }

        template< typename From, typename Symbol >
        using ScopedSymbolTable = llvm::ScopedHashTableScope< From, Symbol >;

        using TypeDefsScope      = ScopedSymbolTable< const clang::TypedefDecl *, hl::TypeDefOp >;
        using TypeDeclsScope     = ScopedSymbolTable< const clang::TypeDecl *, hl::TypeDeclOp >;
        using EnumDeclsScope     = ScopedSymbolTable< const clang::EnumDecl *, hl::EnumDeclOp >;
        using EnumConstantsScope = ScopedSymbolTable< const clang::EnumConstantDecl *, hl::EnumConstantOp >;
        using FunctionsScope     = ScopedSymbolTable< const clang::FunctionDecl *, hl::FuncOp >;
        using VariablesScope     = ScopedSymbolTable< const clang::VarDecl *, Value >;

        struct CodegenScope {
            TypeDefsScope      typedefs;
            TypeDeclsScope     typedecls;
            EnumDeclsScope     enumdecls;
            EnumConstantsScope enumconsts;
            FunctionsScope     funcdecls;
            VariablesScope     globs;
        };

        bool verify_module() const { return mlir::verify(_module.get()).succeeded(); }

        operation get_global_value(string_ref name) {
            return _cgctx->get_global_value(name);
        }

        mlir_value get_global_value(const clang::Decl *decl) {
            return _cgctx->get_global_value(decl);
        }

        const std::vector< clang::GlobalDecl >& default_methods_to_emit() const {
            return _cgctx->default_methods_to_emit;
        }

        const std::vector< clang::GlobalDecl >& deferred_decls_to_emit() const {
            return _cgctx->deferred_decls_to_emit;
        }

        const std::vector< const clang::CXXRecordDecl * >& deferred_vtables() const {
            return _cgctx->deferred_vtables;
        }

        const std::map< string_ref, clang::GlobalDecl >& deferred_decls() const {
            return _cgctx->deferred_decls;
        }

        std::vector< clang::GlobalDecl >&& receive_deferred_decls_to_emit() {
            std::vector< clang::GlobalDecl > current;
            current.swap(_cgctx->deferred_decls_to_emit);
            return std::move(current);
        }

        mlir_type convert(qual_type type) { return _visitor->Visit(type); }

    private:

        void setup_codegen(acontext_t &actx) {
            if (_cgctx)
                return;

            // TODO(Heno): fix module location
            _module = { vast_module::create(mlir::UnknownLoc::get(_mctx)) };

            _cgctx = std::make_unique< CodeGenContext >(*_mctx, actx, _module);

            _scope = std::unique_ptr< CodegenScope >( new CodegenScope{
                .typedefs   = _cgctx->typedefs,
                .typedecls  = _cgctx->typedecls,
                .enumdecls  = _cgctx->enumdecls,
                .enumconsts = _cgctx->enumconsts,
                .funcdecls  = _cgctx->funcdecls,
                .globs      = _cgctx->vars
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

        mcontext_t *_mctx;
        MetaGenerator &_meta;

        std::unique_ptr< CodeGenContext > _cgctx;
        std::unique_ptr< CodegenScope >   _scope;
        std::unique_ptr< CodeGenVisitor > _visitor;

        owning_module_ref _module;
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

        DefaultCodeGen(acontext_t *actx, mcontext_t *mctx)
            : meta(actx, mctx), codegen(actx, mctx, meta)
        {}

        owning_module_ref emit_module(clang::ASTUnit *unit) {
            return codegen.emit_module(unit);
        }

        owning_module_ref emit_module(clang::Decl *decl) {
            return codegen.emit_module(decl);
        }

        bool verify_module() const { return codegen.verify_module(); }

        owning_module_ref freeze() { return codegen.freeze(); }

        mlir_type convert(qual_type type) { return codegen.convert(type); }

        operation build_function_prototype(clang::GlobalDecl decl, mlir_type fty) {
            return codegen.build_function_prototype(decl, fty);
        }

        operation get_global_value(string_ref name) {
            return codegen.get_global_value(name);
        }

        mlir_value get_global_value(const clang::Decl *decl) {
            return codegen.get_global_value(decl);
        }

        const std::vector< clang::GlobalDecl >& default_methods_to_emit() const {
            return codegen.default_methods_to_emit();
        }

        const std::vector< clang::GlobalDecl >& deferred_decls_to_emit() const {
            return codegen.deferred_decls_to_emit();
        }

        const std::vector< const clang::CXXRecordDecl * >& deferred_vtables() const {
            return codegen.deferred_vtables();
        }

        const std::map< string_ref, clang::GlobalDecl >& deferred_decls() const {
            return codegen.deferred_decls();
        }

        std::vector< clang::GlobalDecl >&& receive_deferred_decls_to_emit() {
            return codegen.receive_deferred_decls_to_emit();
        }

        MetaGenerator meta;
        CodeGenBase< Visitor > codegen;
    };

    using CodeGenWithMetaIDs = DefaultCodeGen< DefaultCodeGenVisitorConfig, IDMetaGenerator >;

} // namespace vast::cg
