// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    struct visitor_view
    {
        explicit visitor_view(visitor_base &visitor) : visitor(visitor) {}

        decltype(auto) visit(const clang_decl *decl) { visitor.visit(decl); }
        decltype(auto) visit(const clang_stmt *stmt) { visitor.visit(stmt); }
        decltype(auto) visit(const clang_type *type) { visitor.visit(type); }
        decltype(auto) visit(clang_qual_type ty) { visitor.visit(ty); }
        decltype(auto) visit(const clang_attr *attr) { visitor.visit(attr); }

        decltype(auto) visit(const clang_function_type *fty, bool is_variadic) {
            return visitor.visit(fty, is_variadic);
        }

        decltype(auto) visit_as_lvalue_type(clang_qual_type ty) {
            return visitor.visit_as_lvalue_type(ty);
        }

      private:
        visitor_base &visitor;
    };

} // namespace vast::cg
