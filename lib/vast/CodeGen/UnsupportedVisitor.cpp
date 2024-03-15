// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/UnsupportedVisitor.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"
#include "vast/Dialect/Unsupported/UnsupportedAttributes.hpp"

#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    auto try_make_body_builder(visitor_view visitor, const clang_decl *decl)
        -> std::optional< BuilderCallBackFn >
    {
        auto callback = [&] (auto body) {
            return [&, body] (auto &bld, auto loc) { visitor.visit(body); };
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
            return [&visitor, ctx] (auto &bld, auto loc) {
                for (auto child : ctx->decls()) {
                    visitor.visit(child);
                }
            };
        }

        return std::nullopt;
    }

    std::string decl_name(const clang_decl * decl) {
        std::stringstream ss;
        ss << decl->getDeclKindName();
        if (auto named = dyn_cast< clang::NamedDecl >(decl)) {
            ss << "::" << get_namespaced_decl_name(named);
        }
        return ss.str();
    }

    operation unsup_decl_visitor::visit(const clang_decl *decl) {
        auto op = self.compose< unsup::UnsupportedDecl >()
            .bind(self.location(decl))
            .bind(decl_name(decl))
            .bind_if_valid(try_make_body_builder(self, decl))
            .freeze();

        if (decl->hasAttrs()) {
            mlir::NamedAttrList attrs = op->getAttrs();
            for (auto attr : decl->getAttrs()) {
                attrs.append(attr->getSpelling(), self.visit(attr));
            }
            op->setAttrs(attrs);
        }

        return op;
    }

    std::vector< BuilderCallBackFn > unsup_stmt_visitor::make_children(const clang_stmt *stmt) {
        std::vector< BuilderCallBackFn > children;
        for (auto ch : stmt->children()) {
            // For each subexpression, the unsupported operation holds a region.
            // Last value of the region is an operand of the expression.
            children.push_back([this, ch](auto &bld, auto loc) {
                self.visit(ch);
            });
        }
        return children;
    }

    mlir_type unsup_stmt_visitor::return_type(const clang_stmt *stmt) {
        auto expr = mlir::dyn_cast_or_null< clang_expr >(stmt);
        return expr ? self.visit(expr->getType()) : mlir_type();
    }

    operation unsup_stmt_visitor::visit(const clang_stmt *stmt) {
        auto rty = return_type(stmt);
        return self.builder().create< unsup::UnsupportedStmt >(
            self.location(stmt), stmt->getStmtClassName(), rty, make_children(stmt)
        );
    }

    mlir_type unsup_type_visitor::visit(const clang_type *type) {
        return unsup::UnsupportedType::get(&self.mcontext(), type->getTypeClassName());
    }

    mlir_attr unsup_attr_visitor::visit(const clang_attr *attr) {
        std::string spelling(attr->getSpelling());
        return unsup::UnsupportedAttr::get(&self.mcontext(), spelling);
    }

} // namespace vast::cg
