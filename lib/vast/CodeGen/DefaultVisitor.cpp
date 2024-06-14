// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultVisitor.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"

#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    operation default_visitor::visit(const clang_decl *decl, scope_context &scope) {
        default_decl_visitor visitor(mctx, bld, self, scope);
        visitor.emit_strict_function_return = emit_strict_function_return;
        visitor.missing_return_policy = missing_return_policy;
        return visitor.visit(decl);
    }

    operation default_visitor::visit(const clang_stmt *stmt, scope_context &scope) {
        default_stmt_visitor visitor(mctx, bld, self, scope);
        return visitor.visit(stmt);
    }

    mlir_type default_visitor::visit(const clang_type *type, scope_context &scope) {
        default_type_visitor visitor(mctx, bld, self, scope);
        return visitor.visit(type);
    }

    mlir_type default_visitor::visit(clang_qual_type type, scope_context &scope) {
        default_type_visitor visitor(mctx, bld, self, scope);
        return visitor.visit(type);
    }

    mlir_attr default_visitor::visit(const clang_attr *attr, scope_context &scope) {
        default_attr_visitor visitor(mctx, bld, self, scope);
        return visitor.visit(attr);
    }

    operation default_visitor::visit_prototype(const clang_function *decl, scope_context &scope) {
        default_decl_visitor visitor(mctx, bld, self, scope);
        return visitor.visit_prototype(decl);
    }

    std::optional< loc_t > default_visitor::location(const clang_decl *decl) { return mg->location(decl); }
    std::optional< loc_t > default_visitor::location(const clang_stmt *stmt) { return mg->location(stmt); }
    std::optional< loc_t > default_visitor::location(const clang_expr *expr) { return mg->location(expr); }

    std::optional< symbol_name > default_visitor::symbol(clang_global decl) { return sg->symbol(decl); }
    std::optional< symbol_name > default_visitor::symbol(const clang_decl_ref_expr *decl) { return sg->symbol(decl); }

} // namespace vast::cg
