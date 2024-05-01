// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/UnsupportedVisitor.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"
#include "vast/Dialect/Unsupported/UnsupportedAttributes.hpp"

#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    auto try_make_body_builder(visitor_view visitor, const clang_decl *decl, scope_context &scope)
        -> std::optional< BuilderCallBackFn >
    {
        auto callback = [&] (auto body) {
            return [&, body] (auto &bld, auto loc) { visitor.visit(body, scope); };
        };

        #define VAST_UNSUPPORTED_DECL_BODY_CALLBACK(type, body) \
            if (auto d = mlir::dyn_cast< clang::type >(decl)) { \
                return callback(d->get##body()); \
            }

        VAST_UNSUPPORTED_DECL_BODY_CALLBACK(StaticAssertDecl, AssertExpr)
        VAST_UNSUPPORTED_DECL_BODY_CALLBACK(BlockDecl, Body)
        VAST_UNSUPPORTED_DECL_BODY_CALLBACK(BindingDecl, Binding)
        VAST_UNSUPPORTED_DECL_BODY_CALLBACK(CapturedDecl, Body)
        VAST_UNSUPPORTED_DECL_BODY_CALLBACK(NamespaceAliasDecl, Namespace)
        VAST_UNSUPPORTED_DECL_BODY_CALLBACK(UsingDecl, UnderlyingDecl)
        VAST_UNSUPPORTED_DECL_BODY_CALLBACK(UsingShadowDecl, TargetDecl)

        #undef VAST_UNSUPPORTED_DECL_BODY_CALLBACK

        if (auto ctx = mlir::dyn_cast< clang::DeclContext >(decl)) {
            return [&visitor, ctx, &scope] (auto &bld, auto loc) {
                for (auto child : ctx->decls()) {
                    visitor.visit(child, scope);
                }
            };
        }

        return std::nullopt;
    }

    std::string decl_name(const clang_decl *decl) {
        std::stringstream ss;
        ss << decl->getDeclKindName();
        if (auto named = dyn_cast< clang::NamedDecl >(decl)) {
            ss << "::" << get_namespaced_decl_name(named);
        }
        return ss.str();
    }

    operation unsup_decl_visitor::visit(const clang_decl *decl, scope_context &scope) {
        auto op = bld.compose< unsup::UnsupportedDecl >()
            .bind(self.location(decl))
            .bind(decl_name(decl))
            .bind_if_valid(try_make_body_builder(self, decl, scope))
            .freeze();

        if (decl->hasAttrs()) {
            mlir::NamedAttrList attrs = op->getAttrs();
            for (auto attr : decl->getAttrs()) {
                attrs.append(attr->getSpelling(), self.visit(attr, scope));
            }
            op->setAttrs(attrs);
        }

        return op;
    }

    std::vector< BuilderCallBackFn > unsup_stmt_visitor::make_children(
        const clang_stmt *stmt, scope_context &scope
    ) {
        std::vector< BuilderCallBackFn > children;
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
            self.location(stmt), stmt->getStmtClassName(), rty, make_children(stmt, scope)
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
