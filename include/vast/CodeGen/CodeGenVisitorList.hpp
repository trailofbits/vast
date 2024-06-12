// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include <type_traits>

namespace vast::cg {

    struct visitor_list
    {
        struct node;
        using node_ptr = std::shared_ptr< node >;

        struct node : visitor_base
        {
          protected:
            struct node_tag_t {};

            constexpr static node_tag_t node_tag{};

          public:
            node(node_tag_t) : next(nullptr) {}

            virtual ~node() = default;
            node_ptr next;
        };

        node_ptr head;
        node_ptr tail;
    };

    struct fallthrough_node_proxy : visitor_list::node
    {
        using base = visitor_list::node;
        using base::base;

        virtual ~fallthrough_node_proxy() = default;

        static visitor_list::node_ptr make() {
            return std::make_shared< fallthrough_node_proxy >(base::node_tag);
        }

        operation visit(const clang_decl *decl, scope_context &scope) override {
            return next->visit(decl, scope);
        }

        operation visit(const clang_stmt *stmt, scope_context &scope) override {
            return next->visit(stmt, scope);
        }

        mlir_type visit(const clang_type *type, scope_context &scope) override {
            return next->visit(type, scope);
        }

        mlir_type visit(clang_qual_type type, scope_context &scope) override {
            return next->visit(type, scope);
        }

        mlir_attr visit(const clang_attr *attr, scope_context &scope) override {
            return next->visit(attr, scope);
        }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            return next->visit_prototype(decl, scope);
        }

        std::optional< loc_t > location(const clang_decl *decl) override {
            return next->location(decl);
        }

        std::optional< loc_t > location(const clang_stmt *stmt) override {
            return next->location(stmt);
        }

        std::optional< loc_t > location(const clang_expr *expr) override {
            return next->location(expr);
        }

        std::optional< symbol_name > symbol(clang_global decl) override {
            return next->symbol(decl);
        }

        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override {
            return next->symbol(decl);
        }
    };

} // namespace vast::cg
