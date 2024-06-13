// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"

namespace vast::cg
{
    template< typename func_t >
    concept defered_visitor_builder = std::invocable< func_t, visitor_list::node & >;

    template< typename func_t >
    concept visitor_builder = std::invocable< func_t >;

    struct through_proxy : visitor_list::node
    {
        using base = visitor_list::node;

        std::shared_ptr< visitor_base > visitor;

      private:
        struct deferred_visitor_ctor_tag_t {};
        constexpr static deferred_visitor_ctor_tag_t deferred_visitor_ctor_tag{};
      public:

        through_proxy(node_tag_t, std::shared_ptr< visitor_base > &&visitor)
            : base(base::node_tag), visitor(std::move(visitor))
        {}

        through_proxy(deferred_visitor_ctor_tag_t)
            : base(base::node_tag), visitor(nullptr)
        {}

        virtual ~through_proxy() = default;

        // This function defers the construction of the visitor until the
        // visitor list has a head pointer, as the visitor may use the head
        // pointer to create a head view.
        template< defered_visitor_builder visitor_builder_t >
        static visitor_list::node_ptr make(visitor_builder_t &&bld) {
            auto node = std::make_shared< through_proxy >(deferred_visitor_ctor_tag);
            node->visitor = std::invoke(std::forward< visitor_builder_t >(bld), *node);
            return node;
        }

        static visitor_list::node_ptr make(std::shared_ptr< visitor_base > &&visitor) {
            return std::make_shared< through_proxy >(base::node_tag, std::move(visitor));
        }

        template< visitor_builder visitor_builder_t >
        static visitor_list::node_ptr make(visitor_builder_t &&bld) { return make(bld()); }

        template< typename... args_t >
        auto visit_or_pass_through(args_t&&... args) {
            VAST_ASSERT(visitor != nullptr);
            if (auto result = visitor->visit(args...)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->visit(args...);
        }

        operation visit(const clang_stmt *stmt, scope_context &scope) override {
            return visit_or_pass_through(stmt, scope);
        }

        operation visit(const clang_decl *decl, scope_context &scope) override {
            return visit_or_pass_through(decl, scope);
        }

        mlir_type visit(const clang_type *type, scope_context &scope) override {
            return visit_or_pass_through(type, scope);
        }

        mlir_type visit(clang_qual_type ty, scope_context &scope) override {
            return visit_or_pass_through(ty, scope);
        }

        mlir_attr visit(const clang_attr *attr, scope_context &scope) override {
            return visit_or_pass_through(attr, scope);
        }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            VAST_ASSERT(visitor != nullptr);
            if (auto result = visitor->visit_prototype(decl, scope)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->visit_prototype(decl, scope);
        }

        std::optional< loc_t > location_or_pass_through(const auto *decl) const {
            VAST_ASSERT(visitor != nullptr);
            if (auto result = visitor->location(decl)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->location(decl);
        }

        std::optional< loc_t > location(const clang_decl *decl) override {
            return location_or_pass_through(decl);
        }

        std::optional< loc_t > location(const clang_stmt *stmt) override {
            return location_or_pass_through(stmt);
        }

        std::optional< loc_t > location(const clang_expr *expr) override {
            return location_or_pass_through(expr);
        }

        std::optional< symbol_name > symbol_or_pass_through(const auto &decl) const {
            VAST_ASSERT(visitor != nullptr);
            if (auto result = visitor->symbol(decl)) {
                return result;
            }

            VAST_ASSERT(next != nullptr);
            return next->symbol(decl);
        }

        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override {
            return symbol_or_pass_through(decl);
        }

        std::optional< symbol_name > symbol(clang_global decl) override {
            return symbol_or_pass_through(decl);
        }
    };

    struct through_visitor_list : visitor_list {};

    struct empty_through_visitor_list : through_visitor_list {};

    through_visitor_list operator|(through_visitor_list list, visitor_list::node_ptr &&node) {
        if (list.tail == nullptr) {
            VAST_ASSERT(list.head == nullptr);
            list.tail = node;
            list.head = std::move(node);
        } else {
            list.tail->next = node;
            list.tail       = std::move(node);
        }
        return list;
    }

    template< defered_visitor_builder visitor_builder_t >
    through_visitor_list operator|(empty_through_visitor_list, visitor_builder_t &&bld) {
        auto node = through_proxy::make(std::forward< visitor_builder_t >(bld));
        return {node, node};
    }

    through_visitor_list operator|(through_visitor_list list, std::shared_ptr< visitor_base > &&visitor) {
        return list | through_proxy::make(std::move(visitor));
    }

    template< defered_visitor_builder visitor_builder_t >
    through_visitor_list operator|(through_visitor_list list, visitor_builder_t &&bld) {
        return list | through_proxy::make(std::forward< visitor_builder_t >(bld));
    }

    template< visitor_builder visitor_builder_t >
    through_visitor_list operator|(through_visitor_list list, visitor_builder_t &&bld) {
        return list | through_proxy::make(std::forward< visitor_builder_t >(bld));
    }

    template< typename type >
    through_visitor_list operator|(through_visitor_list list, std::optional< type > &&visitor) {
        if (visitor) {
            return list | std::move(*visitor);
        } else {
            return list;
        }
    }

} // namespace vast::cg
