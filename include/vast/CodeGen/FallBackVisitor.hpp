// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

namespace vast::cg
{
    //
    // fallback_visitor
    //
    // Allows to specify chain of fallback visitors in case that first `visitor::visit` is
    // unsuccessful.
    //
    using visitor_base_ptr = std::unique_ptr< visitor_base >;

    struct fallback_visitor : visitor_base
    {
        fallback_visitor(auto &&... visitors)
            : visitors{std::forward< decltype(visitors) >(visitors)...}
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

        mlir_type visit(const clang_function_type *fty, bool is_variadic) override
        {
            return visit_with_fallback(fty, is_variadic);
        }

        std::vector< visitor_base_ptr > visitors;
    };

} // namespace vast::cg
