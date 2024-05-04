// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultVisitor.hpp"

namespace vast::cg
{
    operation default_visitor::visit(const clang_decl *decl, scope_context &scope) {
        default_decl_visitor visitor(bld, self, scope);
        return visitor.visit(decl);
    }

    operation default_visitor::visit(const clang_stmt *stmt, scope_context &scope) {
        default_stmt_visitor visitor(bld, self, scope);
        return visitor.visit(stmt);
    }

    mlir_type default_visitor::visit(const clang_type *type, scope_context &scope) {
        if (auto value = cache.lookup(type)) {
            return value;
        }

        default_type_visitor visitor(bld, self, scope);
        auto result = visitor.visit(type);
        cache.try_emplace(type, result);
        return result;
    }

    mlir_type default_visitor::visit(clang_qual_type type, scope_context &scope) {
        if (auto value = qual_cache.lookup(type)) {
            return value;
        }

        default_type_visitor visitor(bld, self, scope);
        auto result = visitor.visit(type);
        qual_cache.try_emplace(type, result);
        return result;
    }

    mlir_attr default_visitor::visit(const clang_attr *attr, scope_context &scope) {
        default_attr_visitor visitor(bld, self, scope);
        return visitor.visit(attr);
    }

    operation default_visitor::visit_prototype(const clang_function *decl, scope_context &scope) {
        default_decl_visitor visitor(bld, self, scope);
        return visitor.visit_prototype(decl);
    }

} // namespace vast::cg
