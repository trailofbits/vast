// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::cg
{
    operation visitor_view::visit(const clang_decl *decl, scope_context &scope) {
        return visitor.visit(decl, scope);
    }

    operation visitor_view::visit(const clang_stmt *stmt, scope_context &scope) {
        return visitor.visit(stmt, scope);
    }

    mlir_type visitor_view::visit(const clang_type *type, scope_context &scope) {
        return visitor.visit(type, scope);
    }

    mlir_type visitor_view::visit(clang_qual_type ty, scope_context &scope) {
        return visitor.visit(ty, scope);
    }

    std::optional< named_attr > visitor_view::visit(const clang_attr *attr, scope_context &scope) {
        return visitor.visit(attr, scope);
    }

    operation visitor_view::visit_prototype(const clang_function *decl, scope_context &scope) {
        return visitor.visit_prototype(decl, scope);
    }

    operation scoped_visitor_view::visit(const clang_decl *decl) {
        return visitor_view::visit(decl, scope);
    }

    operation scoped_visitor_view::visit(const clang_stmt *stmt) {
        return visitor_view::visit(stmt, scope);
    }

    mlir_type scoped_visitor_view::visit(const clang_type *type) {
        return visitor_view::visit(type, scope);
    }

    mlir_type scoped_visitor_view::visit(clang_qual_type ty) {
        return visitor_view::visit(ty, scope);
    }

    std::optional< named_attr > scoped_visitor_view::visit(const clang_attr *attr) {
        return visitor_view::visit(attr, scope);
    }

    operation scoped_visitor_view::visit_prototype(const clang_function *decl) {
        return visitor_view::visit_prototype(decl, scope);
    }

} // namespace vast::cg
