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

        fallback_visitor(mcontext_t &mctx, meta_generator &mg, symbol_generator &sg, visitor_stack &&visitors)
            : visitor_base(mctx, mg, sg), visitors(std::move(visitors))
        {}

        auto visit_with_fallback(auto &&...tokens) {
            for (auto &visitor : visitors) {
                if (auto result = visitor->visit(std::forward< decltype(tokens) >(tokens)...)) {
                    return result;
                }
            }

            VAST_UNREACHABLE("Vistors chain exhausted. No fallback visitor was able to handle the token.");
        }

        operation visit(const clang_stmt *stmt) override { return visit_with_fallback(stmt); }
        operation visit(const clang_decl *decl) override { return visit_with_fallback(decl); }
        mlir_type visit(const clang_type *type) override { return visit_with_fallback(type); }
        mlir_attr visit(const clang_attr *attr) override { return visit_with_fallback(attr); }
        mlir_type visit(clang_qual_type type) override { return visit_with_fallback(type); }

        operation visit_prototype(const clang_function *decl) override {
            for (auto &visitor : visitors) {
                if (auto result = visitor->visit_prototype(decl)) {
                    return result;
                }
            }

            VAST_UNREACHABLE("Vistors chain exhausted. No fallback visitor was able to handle prototype.");
        }

        visitor_view front() { return visitor_view(*visitors.front()); }

        visitor_stack visitors;
    };

} // namespace vast::cg
