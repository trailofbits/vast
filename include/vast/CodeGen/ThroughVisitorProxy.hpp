// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"

namespace vast::cg
{
    struct through : visitor_node
    {
        using visitor_node::visitor_node;

        virtual ~through() = default;

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

    struct empty_pass_thorugh_list_t {};
    constexpr static empty_pass_thorugh_list_t empty_pass_thorugh_visitor_list{};


    visitor_list operator|(visitor_list list, std::shared_ptr< visitor_base > visitor) {
        return list | through::make< through >(std::move(visitor));
    }

    template< visitor_builder builder_t >
    visitor_list operator|(empty_pass_thorugh_list_t, builder_t &&visitor_builder) {
        return through::make< through >(std::forward< builder_t >(visitor_builder));
    }

} // namespace vast::cg
