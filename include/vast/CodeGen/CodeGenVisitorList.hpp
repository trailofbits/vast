// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include <type_traits>

namespace vast::cg {

    struct visitor_list;
    struct visitor_list_node;

    using visitor_list_ptr = std::shared_ptr< visitor_list >;
    using visitor_node_ptr = std::shared_ptr< visitor_list_node >;

    struct visitor_list_node : visitor_base {
        visitor_node_ptr next;
    };

    template< typename visitor >
    struct visitor_list_node_adaptor : visitor_list_node, visitor {
        using base_visitor = visitor;

        template< typename... args_t >
        visitor_list_node_adaptor(args_t&&... args) : visitor(std::forward< args_t >(args)...) {}

        operation visit(const clang_decl *decl, scope_context &scope) override { return visitor::visit(decl, scope); }
        operation visit(const clang_stmt *stmt, scope_context &scope) override { return visitor::visit(stmt, scope); }
        mlir_type visit(const clang_type *type, scope_context &scope) override { return visitor::visit(type, scope); }
        mlir_type visit(clang_qual_type type, scope_context &scope)   override { return visitor::visit(type, scope); }
        mlir_attr visit(const clang_attr *attr, scope_context &scope) override { return visitor::visit(attr, scope); }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            return visitor::visit_prototype(decl, scope);
        }

        std::optional< loc_t > location(const clang_decl *decl) override { return visitor::location(decl); }
        std::optional< loc_t > location(const clang_stmt *stmt) override { return visitor::location(stmt); }
        std::optional< loc_t > location(const clang_expr *expr) override { return visitor::location(expr); }

        std::optional< symbol_name > symbol(clang_global decl) override { return visitor::symbol(decl); }
        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override { return visitor::symbol(decl); }
    };

    template< typename node_type >
    struct list
    {
        std::shared_ptr< node_type > head;
        std::shared_ptr< node_type > tail;

        struct iterator
        {
            using self_type         = iterator;
            using value_type        = node_type;
            using reference         = node_type &;
            using pointer           = node_type *;
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = ptrdiff_t;

            explicit iterator(std::shared_ptr< node_type > ptr) : ptr(ptr) {}

            self_type operator++() {
                if (ptr) {
                    ptr = ptr->next;
                }
                return *this;
            }

            self_type operator++(int) {
                self_type i = *this;
                if (ptr) {
                    ptr = ptr->next;
                }
                return i;
            }

            reference operator*() { return *ptr; }
            pointer operator->() { return ptr.get(); }

            bool operator==(const self_type &other) const { return ptr == other.ptr; }
            bool operator!=(const self_type &other) const { return ptr != other.ptr; }

          private:
            std::shared_ptr< node_type > ptr;
        };

        iterator begin() { return iterator(head); }
        iterator end() { return iterator(nullptr); }
    };

    struct visitor_list : list< visitor_list_node >, visitor_base {
        using list = list< visitor_list_node >;

        using list::head;
        using list::tail;

        operation visit(const clang_decl *decl, scope_context &scope) override { return head->visit(decl, scope); }
        operation visit(const clang_stmt *stmt, scope_context &scope) override { return head->visit(stmt, scope); }
        mlir_type visit(const clang_type *type, scope_context &scope) override { return head->visit(type, scope); }
        mlir_type visit(clang_qual_type type, scope_context &scope)   override { return head->visit(type, scope); }
        mlir_attr visit(const clang_attr *attr, scope_context &scope) override { return head->visit(attr, scope); }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            return head->visit_prototype(decl, scope);
        }

        std::optional< loc_t > location(const clang_decl *decl) override { return head->location(decl); }
        std::optional< loc_t > location(const clang_stmt *stmt) override { return head->location(stmt); }
        std::optional< loc_t > location(const clang_expr *expr) override { return head->location(expr); }

        std::optional< symbol_name > symbol(clang_global decl) override { return head->symbol(decl); }
        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override { return head->symbol(decl); }
    };

    struct fallthrough_list_node : visitor_list_node {
        operation visit(const clang_decl *decl, scope_context &scope) override { return next->visit(decl, scope); }
        operation visit(const clang_stmt *stmt, scope_context &scope) override { return next->visit(stmt, scope); }
        mlir_type visit(const clang_type *type, scope_context &scope) override { return next->visit(type, scope); }
        mlir_type visit(clang_qual_type type, scope_context &scope)   override { return next->visit(type, scope); }
        mlir_attr visit(const clang_attr *attr, scope_context &scope) override { return next->visit(attr, scope); }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            return next->visit_prototype(decl, scope);
        }

        std::optional< loc_t > location(const clang_decl *decl) override { return next->location(decl); }
        std::optional< loc_t > location(const clang_stmt *stmt) override { return next->location(stmt); }
        std::optional< loc_t > location(const clang_expr *expr) override { return next->location(expr); }

        std::optional< symbol_name > symbol(clang_global decl) override { return next->symbol(decl); }
        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override { return next->symbol(decl); }
    };


    template< typename visitor >
    struct try_or_through_list_node : visitor_list_node_adaptor< visitor > {
        using base = visitor_list_node_adaptor< visitor >;
        using base_visitor = base::base_visitor;

        using base::next;

        template< typename... args_t >
        try_or_through_list_node(args_t&&... args) : base(std::forward< args_t >(args)...) {}

        template< typename... args_t >
        auto try_visit_or_pass(args_t &&... args) {
            if (auto result = base_visitor::visit(args...)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->visit(std::forward< args_t >(args)...);
        }

        operation visit(const clang_decl *decl, scope_context &scope) override { return try_visit_or_pass(decl, scope); }
        operation visit(const clang_stmt *stmt, scope_context &scope) override { return try_visit_or_pass(stmt, scope); }
        mlir_type visit(const clang_type *type, scope_context &scope) override { return try_visit_or_pass(type, scope); }
        mlir_type visit(clang_qual_type type, scope_context &scope)   override { return try_visit_or_pass(type, scope); }
        mlir_attr visit(const clang_attr *attr, scope_context &scope) override { return try_visit_or_pass(attr, scope); }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            if (auto result = base_visitor::visit_prototype(decl, scope)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->visit_prototype(decl, scope);
        }

        template< typename... args_t >
        auto try_location_or_pass(args_t &&... args) {
            if (auto result = base_visitor::location(args...)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->location(std::forward< args_t >(args)...);
        }

        std::optional< loc_t > location(const clang_decl *decl) override { return try_location_or_pass(decl); }
        std::optional< loc_t > location(const clang_stmt *stmt) override { return try_location_or_pass(stmt); }
        std::optional< loc_t > location(const clang_expr *expr) override { return try_location_or_pass(expr); }

        template< typename... args_t >
        auto try_symbol_or_pass(args_t &&... args) {
            if (auto result = base_visitor::symbol(args...)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->symbol(std::forward< args_t >(args)...);
        }

        std::optional< symbol_name > symbol(clang_global decl) override { return try_symbol_or_pass(decl); }
        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override { return try_symbol_or_pass(decl); }
    };

    template< typename visitor >
    using visitor_node_adaptor_ptr = std::shared_ptr< try_or_through_list_node< visitor > >;


    template< typename visitor, typename ...args_t >
    auto as_node(args_t &&... args) -> visitor_node_adaptor_ptr< visitor > {
        return visitor_node_adaptor_ptr< visitor >(std::forward< args_t >(args)...);
    }

    template< typename proxy, typename ...args_t >
    requires std::derived_from< proxy, visitor_list_node >
    auto as_node(args_t &&... args) -> visitor_node_ptr {
        return std::make_shared< proxy >(std::forward< args_t >(args)...);
    }

    using node_with_list_ref_wrap = std::function<
        std::shared_ptr< visitor_list_node >(visitor_list& list)
    >;

    template< typename proxy, typename... args_t >
    requires std::derived_from< proxy, visitor_list_node >
    node_with_list_ref_wrap as_node_with_list_ref(args_t &&...args) {
        return [&args...](visitor_list& list) {
            return std::make_shared< proxy >(
                static_cast< visitor_base& >(list), std::forward<args_t>(args)...
            );
        };
    }

    template< typename visitor, typename... args_t >
    node_with_list_ref_wrap as_node_with_list_ref(args_t &&... args) {
        return as_node_with_list_ref< try_or_through_list_node< visitor > >(
            std::forward<args_t>(args)...
        );
    }

    template< typename node_type >
    std::optional< node_type > optional(bool enable, node_type &&node) {
        return enable ? std::make_optional(std::forward< node_type >(node)) : std::nullopt;
    }

    visitor_list_ptr operator|(visitor_list_ptr &&list, visitor_node_ptr &&node);

    template< typename node_type >
    visitor_list_ptr operator|(visitor_list_ptr &&list, std::optional< node_type > &&node) {
        if (node)
            return std::move(list) | std::move(node.value());
        return std::move(list);
    }

    visitor_list_ptr operator|(visitor_list_ptr &&list, node_with_list_ref_wrap &&wrap);


} // namespace vast::cg
