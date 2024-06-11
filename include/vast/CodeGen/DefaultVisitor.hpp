// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/ClangVisitorBase.hpp"
#include "vast/CodeGen/DefaultAttrVisitor.hpp"
#include "vast/CodeGen/DefaultDeclVisitor.hpp"
#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/DefaultTypeVisitor.hpp"

#include "vast/CodeGen/CodeGenFunction.hpp"

namespace vast::cg
{
    struct default_visitor final : visitor_base
    {
        default_visitor(
              mcontext_t &mctx
            , codegen_builder &bld
            , visitor_view self
            , std::shared_ptr< meta_generator > mg
            , std::shared_ptr< symbol_generator > sg
        )
            : mctx(mctx)
            , bld(bld)
            , self(self)
            , mg(std::move(mg))
            , sg(std::move(sg))
        {}

        operation visit_with_attrs(const clang_decl *decl, scope_context &scope);
        operation visit_decl_attrs(operation op, const clang_decl *decl, scope_context &scope);

        operation visit(const clang_decl *decl, scope_context &scope) override;
        operation visit(const clang_stmt *stmt, scope_context &scope) override;
        mlir_type visit(const clang_type *type, scope_context &scope) override;
        mlir_type visit(clang_qual_type type, scope_context &scope) override;
        mlir_attr visit(const clang_attr *attr, scope_context &scope) override;

        operation visit_prototype(const clang_function *decl, scope_context &scope) override;

        mlir_type visit_type(auto type, auto& cache, scope_context& scope);

        std::optional< loc_t > location(const clang_decl *) override;
        std::optional< loc_t > location(const clang_stmt *) override;
        std::optional< loc_t > location(const clang_expr *) override;

        std::optional< symbol_name > symbol(clang_global decl) override;
        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override;

        // FIXME: this shouldnt be part of default visitor -- add caching layer instead
        llvm::DenseMap< const clang_type *, mlir_type > cache;
        llvm::DenseMap< clang_qual_type, mlir_type > qual_cache;

        // FIXME: This should be store on single location
        bool emit_strict_function_return;
        missing_return_policy missing_return_policy;

        mcontext_t &mctx;
        codegen_builder &bld;
        visitor_view self;

        std::shared_ptr< meta_generator > mg;
        std::shared_ptr< symbol_generator > sg;

    };

    mlir_type default_visitor::visit_type(auto type, auto& cache, scope_context& scope) {
        if (auto value = cache.lookup(type)) {
            return value;
        }

        default_type_visitor visitor(mctx, bld, self, scope);
        if (auto result = visitor.visit(type)) {
            cache.try_emplace(type, result);
            return result;
        } else {
            return {};
        }
    }

} // namespace vast::cg
