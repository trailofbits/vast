// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitor.hpp"

namespace vast::cg {

    namespace detail
    {
        template< typename visitor_base >
        struct visitor_view
        {
            explicit visitor_view(visitor_base &visitor) : visitor(visitor) {}

            operation visit(const clang_decl *decl) { return visitor.visit(decl); }
            operation visit(const clang_stmt *stmt) { return visitor.visit(stmt); }
            mlir_type visit(const clang_type *type) { return visitor.visit(type); }
            mlir_type visit(clang_qual_type ty)     { return visitor.visit(ty); }
            mlir_attr visit(const clang_attr *attr) { return visitor.visit(attr); }

            loc_t location(const auto *node) const { return visitor.location(node); }

            mcontext_t& mcontext() { return visitor.mcontext(); }
            const mcontext_t& mcontext() const { return visitor.mcontext(); }

          protected:
            visitor_base &visitor;
        };
    } // namespace detail


    using base_visitor_view = detail::visitor_view< visitor_base >;

    struct codegen_visitor_view : detail::visitor_view< codegen_visitor_base >
    {
        using detail::visitor_view< codegen_visitor_base >::visitor_view;

        mlir_type visit(const clang_function_type *fty, bool is_variadic) {
            return visitor.visit(fty, is_variadic);
        }

        mlir_type visit_as_lvalue_type(clang_qual_type ty) {
            return visitor.visit_as_lvalue_type(ty);
        }
    };

} // namespace vast::cg
