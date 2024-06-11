// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

namespace vast::cg {

    struct visitor_base;

    struct visitor_view
    {
        explicit visitor_view(visitor_base &visitor) : visitor(visitor) {}

        operation visit(const clang_decl *decl, scope_context &scope);
        operation visit(const clang_stmt *stmt, scope_context &scope);
        mlir_type visit(const clang_type *type, scope_context &scope);
        mlir_type visit(clang_qual_type ty, scope_context &scope);
        mlir_attr visit(const clang_attr *attr, scope_context &scope);

        operation visit_prototype(const clang_function *decl, scope_context &scope);

        std::optional< loc_t > location(const auto *node) const;

        std::optional< symbol_name > symbol(auto &&decl);

        visitor_base *raw() { return &visitor; }

      protected:
        visitor_base &visitor;
    };


    struct scoped_visitor_view : visitor_view
    {
        explicit scoped_visitor_view(visitor_base &visitor, scope_context &scope)
            : visitor_view(visitor), scope(scope)
        {}

        explicit scoped_visitor_view(visitor_view view, scope_context &scope)
            : visitor_view(std::move(view)), scope(scope)
        {}

        operation visit(const clang_decl *decl);
        operation visit(const clang_stmt *stmt);
        mlir_type visit(const clang_type *type);
        mlir_type visit(clang_qual_type ty);
        mlir_attr visit(const clang_attr *attr);

        operation visit_prototype(const clang_function *decl);

        scope_context &scope;
    };

    //
    // Classes derived from `visitor_base` are used to visit clang AST nodes
    //
    struct visitor_base
    {
        virtual ~visitor_base() = default;

        virtual operation visit(const clang_decl *, scope_context &scope) = 0;
        virtual operation visit(const clang_stmt *, scope_context &scope) = 0;
        virtual mlir_type visit(const clang_type *, scope_context &scope) = 0;
        virtual mlir_type visit(clang_qual_type, scope_context &scope)    = 0;
        virtual mlir_attr visit(const clang_attr *, scope_context &scope) = 0;

        virtual operation visit_prototype(const clang_function *decl, scope_context &scope) = 0;

        virtual std::optional< loc_t > location(const clang_decl *) { return std::nullopt; }
        virtual std::optional< loc_t > location(const clang_stmt *) { return std::nullopt; }
        virtual std::optional< loc_t > location(const clang_expr *) { return std::nullopt; }

        virtual std::optional< symbol_name > symbol(clang_global decl) { return std::nullopt; }
        virtual std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) { return std::nullopt; }
    };

    std::optional< loc_t > visitor_view::location(const auto *node) const {
        return visitor.location(node);
    }

    std::optional< symbol_name > visitor_view::symbol(auto &&decl) {
        return visitor.symbol(std::forward< decltype(decl) >(decl));
    }

} // namespace vast::cg
