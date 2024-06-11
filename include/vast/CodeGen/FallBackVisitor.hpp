// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg
{
    //
    // fallback_visitor
    //
    // Allows to specify chain of fallback visitors in case that first `visitor::visit` is
    // unsuccessful.
    //
    struct fallback_visitor : visitor_base
    {
        using visitor_stack = std::vector< visitor_base_ptr >;

        auto visit_with_fallback(auto &&...tokens) {
            for (auto &visitor : visitors) {
                if (auto result = visitor->visit(tokens...)) {
                    return result;
                }
            }

            VAST_UNREACHABLE("Vistors chain exhausted. No fallback visitor was able to handle the token.");
        }

        template< typename visitor_t >
        std::optional< visitor_t > find() {
            for (auto &visitor : visitors) {
                if (auto result = dynamic_cast< visitor_t* >(visitor.get())) {
                    return *result;
                }
            }

            return std::nullopt;
        }

        void push(visitor_base_ptr &&visitor) {
            visitors.push_back(std::move(visitor));
        }

        operation visit(const clang_stmt *stmt, scope_context &scope) override {
            return visit_with_fallback(stmt, scope);
        }

        operation visit(const clang_decl *decl, scope_context &scope) override {
            return visit_with_fallback(decl, scope);
        }

        mlir_type visit(const clang_type *type, scope_context &scope) override {
            return visit_with_fallback(type, scope);
        }

        mlir_attr visit(const clang_attr *attr, scope_context &scope) override {
            return visit_with_fallback(attr, scope);
        }

        mlir_type visit(clang_qual_type type, scope_context &scope) override {
            return visit_with_fallback(type, scope);
        }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            for (auto &visitor : visitors) {
                if (auto result = visitor->visit_prototype(decl, scope)) {
                    return result;
                }
            }

            VAST_UNREACHABLE("Vistors chain exhausted. No fallback visitor was able to handle prototype.");
        }

        std::optional< loc_t > location_with_fallback(const auto *node) {
            for (auto &visitor : visitors) {
                if (auto result = visitor->location(node)) {
                    return result;
                }
            }

            VAST_UNREACHABLE("Vistors chain exhausted. No fallback visitor was able to handle location.");
        }

        std::optional< loc_t > location(const clang_decl *decl) override {
            return location_with_fallback(decl);
        }

        std::optional< loc_t > location(const clang_stmt *stmt) override {
            return location_with_fallback(stmt);
        }

        std::optional< loc_t > location(const clang_expr *expr) override {
            return location_with_fallback(expr);
        }

        std::optional< symbol_name > symbol_with_fallback(auto node) {
            for (auto &visitor : visitors) {
                if (auto result = visitor->symbol(node)) {
                    return result;
                }
            }

            VAST_UNREACHABLE("Vistors chain exhausted. No fallback visitor was able to handle symbol.");
        }

        std::optional< symbol_name > symbol(clang_global decl) override {
            return symbol_with_fallback(decl);
        }
        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override {
            return symbol_with_fallback(decl);
        }


        visitor_view front() { return visitor_view(*visitors.front()); }

        visitor_stack visitors;
    };

} // namespace vast::cg
