// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/DefaultAttrVisitor.hpp"
#include "vast/CodeGen/DefaultDeclVisitor.hpp"
#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/DefaultTypeVisitor.hpp"

namespace vast::cg
{
    struct default_visitor final : visitor_base
    {
        default_visitor(
              mcontext_t &mctx
            , codegen_builder &bld
            , meta_generator &mg
            , symbol_generator &sg
            , visitor_view self
        )
            : visitor_base(mctx, mg, sg, self.options()), bld(bld), self(self)
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

        llvm::DenseMap< const clang_type *, mlir_type > cache;
        llvm::DenseMap< clang_qual_type, mlir_type > qual_cache;

        codegen_builder &bld;
        visitor_view self;
    };

    mlir_type default_visitor::visit_type(auto type, auto& cache, scope_context& scope) {
        if (auto value = cache.lookup(type)) {
            return value;
        }

        default_type_visitor visitor(bld, self, scope);
        if (auto result = visitor.visit(type)) {
            cache.try_emplace(type, result);
            return result;
        } else {
            return {};
        }
    }

} // namespace vast::cg
