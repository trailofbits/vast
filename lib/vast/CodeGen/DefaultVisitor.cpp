// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultVisitor.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"

#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    operation default_visitor::visit_with_attrs(const clang_decl *decl, scope_context &scope) {
        default_decl_visitor visitor(mctx, bld, self, scope);
        visitor.emit_strict_function_return = emit_strict_function_return;
        visitor.missing_return_policy = missing_return_policy;
        if (auto op = visitor.visit(decl)) {
            return visit_decl_attrs(op, decl, scope);
        }

        return {};
    }

    using excluded_attr_list = util::type_list<
          clang::WeakAttr
        , clang::SelectAnyAttr
        , clang::CUDAGlobalAttr
    >;

    operation default_visitor::visit_decl_attrs(
        operation op, const clang_decl *decl, scope_context &scope
    ) {
        if (decl->hasAttrs()) {
            mlir::NamedAttrList attrs = op->getAttrs();
            for (auto attr : exclude_attrs< excluded_attr_list >(decl->getAttrs())) {
                auto visited = self.visit(attr, scope);

                // All attributes in unsupported dialect have the same name
                // TODO (#613): Move this to unsupported dialect
                auto is_unsup = mlir::isa< unsup::UnsupportedDialect >(visited.getDialect());
                auto key =
                    is_unsup ? attr->getSpelling() : visited.getAbstractAttribute().getName();

                attrs.set(key, visited);
            }
            op->setAttrs(attrs);
        }

        return op;
    }

    operation default_visitor::visit(const clang_decl *decl, scope_context &scope) {
        return visit_with_attrs(decl, scope);
    }

    operation default_visitor::visit(const clang_stmt *stmt, scope_context &scope) {
        default_stmt_visitor visitor(mctx, bld, self, scope);
        return visitor.visit(stmt);
    }

    mlir_type default_visitor::visit(const clang_type *type, scope_context &scope) {
        return visit_type(type, cache, scope);
    }

    mlir_type default_visitor::visit(clang_qual_type type, scope_context &scope) {
        return visit_type(type, qual_cache, scope);
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
