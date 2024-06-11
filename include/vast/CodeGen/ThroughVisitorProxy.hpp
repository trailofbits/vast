// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg
{
    struct through_proxy : visitor_base
    {
        through_proxy(std::shared_ptr< visitor_base > &&visitor, std::shared_ptr< visitor_base > &&next)
            : visitor(std::move(visitor)), next(std::move(next))
        {}

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

        static std::shared_ptr< through_proxy > empty() {
            return std::make_shared< through_proxy >(nullptr, nullptr);
        }

        friend std::shared_ptr< through_proxy > operator|(std::shared_ptr< through_proxy > &&lhs, std::shared_ptr< visitor_base > &&rhs) {
            if (lhs->visitor == nullptr) {
                lhs->visitor = std::move(rhs);
                return std::move(lhs);
            }

            if (lhs->next == nullptr) {
                lhs->next = std::move(rhs);
                return std::move(lhs);
            }

            return std::make_shared< through_proxy >(std::move(lhs), std::move(rhs));
        }

    private:
        std::shared_ptr< visitor_base > visitor;
        std::shared_ptr< visitor_base > next;
    };

} // namespace vast::cg
