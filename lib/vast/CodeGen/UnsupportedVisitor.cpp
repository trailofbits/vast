// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/UnsupportedVisitor.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"
#include "vast/Dialect/Unsupported/UnsupportedAttributes.hpp"

#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    std::string decl_name(const clang_decl *decl, visitor_view visitor) {
        std::stringstream ss;
        ss << decl->getDeclKindName();
        if (auto named = dyn_cast< clang::NamedDecl >(decl)) {
            ss << "::" << named->getNameAsString();
        }
        return ss.str();
    }

    operation unsup_decl_visitor::visit(const clang_decl *decl, scope_context &scope) {
        auto op = bld.compose< unsup::UnsupportedDecl >()
            .bind(self.maybe_location(decl))
            .bind(decl_name(decl, self))
            .freeze();

        return op;
    }

    std::vector< builder_callback > unsup_stmt_visitor::make_children(
        const clang_stmt *stmt, scope_context &scope
    ) {
        std::vector< builder_callback > children;
        for (auto ch : stmt->children()) {
            // For each subexpression, the unsupported operation holds a region.
            // Last value of the region is an operand of the expression.
            children.push_back([this, ch, &scope](auto &bld, auto loc) {
                self.visit(ch, scope);
            });
        }
        return children;
    }

    mlir_type unsup_stmt_visitor::return_type(const clang_stmt *stmt, scope_context &scope) {
        auto expr = mlir::dyn_cast_or_null< clang_expr >(stmt);
        return expr ? self.visit(expr->getType(), scope) : mlir_type();
    }

    operation unsup_stmt_visitor::visit(const clang_stmt *stmt, scope_context &scope) {
        auto rty = return_type(stmt, scope);
        return bld.create< unsup::UnsupportedStmt >(
            self.maybe_location(stmt), stmt->getStmtClassName(), rty, make_children(stmt, scope)
        );
    }

    mlir_type unsup_type_visitor::visit(const clang_type *type, scope_context &scope) {
        return unsup::UnsupportedType::get(&self.mcontext(), type->getTypeClassName());
    }

    mlir_type unsup_type_visitor::visit(clang_qual_type type, scope_context &scope) {
        VAST_ASSERT(!type.isNull());
        return visit(type.getTypePtr(), scope);
    }

    mlir_attr unsup_attr_visitor::visit(const clang_attr *attr, scope_context &scope) {
        std::string spelling(attr->getSpelling());
        return unsup::UnsupportedAttr::get(&self.mcontext(), spelling);
    }

} // namespace vast::cg
