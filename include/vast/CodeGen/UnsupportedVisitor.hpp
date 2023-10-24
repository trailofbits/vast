// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"
#include "vast/Dialect/Unsupported/UnsupportedAttributes.hpp"

#include "vast/CodeGen/CodeGen.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenContext.hpp"
#include "vast/CodeGen/CodeGenVisitor.hpp"

#include <functional>

namespace vast::cg {

    template< typename derived_t >
    struct unsup_stmt_visitor
        : stmt_visitor_base< derived_t >
        , visitor_lens< derived_t, unsup_stmt_visitor >
    {
        using lens = visitor_lens< derived_t, unsup_stmt_visitor >;

        using lens::derived;
        using lens::visit;

        using lens::meta_location;

        operation make_unsupported_stmt(auto stmt, mlir_type type = {}) {
            std::vector< BuilderCallBackFn > children;
            for (auto ch : stmt->children()) {
                // For each subexpression, the unsupported operation holds a region.
                // Last value of the region is an operand of the expression.
                children.push_back([this, ch](auto &bld, auto loc) {
                    this->visit(ch);
                });
            }

            return this->template make_operation< unsup::UnsupportedStmt >()
                .bind(meta_location(stmt))
                .bind(stmt->getStmtClassName())
                .bind(type)
                .bind(children)
                .freeze();
        }

        operation Visit(const clang::Stmt *stmt) {
            if (auto expr = mlir::dyn_cast< clang::Expr >(stmt)) {
                return make_unsupported_stmt(expr, visit(expr->getType()));
            }

            return make_unsupported_stmt(stmt);
        }
    };

    template< typename derived_t >
    struct unsup_decl_visitor
        : decl_visitor_base< derived_t >
        , visitor_lens< derived_t, unsup_decl_visitor >
    {
        using lens = visitor_lens< derived_t, unsup_decl_visitor >;

        using lens::derived;
        using lens::context;
        using lens::visit;

        using lens::make_operation;
        using lens::meta_location;

        auto make_body_callback(auto decl) -> std::optional< BuilderCallBackFn > {
            auto callback = [&] (auto body) {
                return [&, body] (auto &bld, auto loc) { visit(body); };
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

            if (auto d = mlir::dyn_cast< clang::DeclContext >(decl)) {
                return [this, d] (auto &bld, auto loc) {
                    for (auto child : d->decls()) {
                        this->visit(child);
                    }
                };
            }

            return std::nullopt;
        };

        std::string decl_name(auto decl) {
            std::stringstream ss;
            ss << decl->getDeclKindName();
            if (auto named = dyn_cast< clang::NamedDecl >(decl)) {
                ss << "::" << context().decl_name(named).str();
            }
            return ss.str();
        }

        operation make_unsupported_decl(auto decl) {
            auto op = this->template make_operation< unsup::UnsupportedDecl >()
                .bind(meta_location(decl)) // location
                .bind(decl_name(decl));    // name

            if (auto callback = make_body_callback(decl)) {
                return std::move(op).bind(callback.value()).freeze(); // body
            }

            return op.freeze();
        }

        operation Visit(const clang::Decl *decl) {
            if (auto op = make_unsupported_decl(decl)) {
                if (decl->hasAttrs()) {
                    mlir::NamedAttrList attrs = op->getAttrs();
                    for (auto attr : decl->getAttrs()) {
                        attrs.append(attr->getSpelling(), visit(attr));
                    }
                    op->setAttrs(attrs);
                }
                return op;
            }
            return {};

        }
    };

    template< typename derived_t >
    struct unsup_type_visitor
        : type_visitor_base< derived_t >
        , visitor_lens< derived_t, unsup_type_visitor >
    {
        using lens = visitor_lens< derived_t, unsup_type_visitor >;
        using lens::mcontext;
        using lens::derived;

        auto make_unsupported_type(auto ty) {
            return derived().template make_type< unsup::UnsupportedType >()
                .bind(&mcontext())
                .bind(ty->getTypeClassName())
                .freeze();
        }

        mlir_type Visit(clang::QualType ty) {
            return ty.isNull() ? Type() : Visit(ty.getTypePtr());
        }

        mlir_type Visit(const clang::Type *ty) {
            return make_unsupported_type(ty);
        }
    };

    template< typename derived_t >
    struct unsup_attr_visitor
        : attr_visitor_base< derived_t >
        , visitor_lens< derived_t, unsup_attr_visitor >
    {
        using lens = visitor_lens< derived_t, unsup_attr_visitor >;

        using lens::mcontext;
        using lens::derived;

        auto make_unsupported_attr(auto attr) {
            std::string spelling(attr->getSpelling());
            return derived().base_builder().template getAttr< unsup::UnsupportedAttr >(spelling);
        }

        mlir_attr Visit(const clang::Attr *attr) {
            return make_unsupported_attr(attr);
        }
    };

    template< typename derived_t >
    struct unsup_visitor
        : unsup_decl_visitor< derived_t >
        , unsup_stmt_visitor< derived_t >
        , unsup_type_visitor< derived_t >
        , unsup_attr_visitor< derived_t >
    {
        using decl_visitor = unsup_decl_visitor< derived_t >;
        using stmt_visitor = unsup_stmt_visitor< derived_t >;
        using type_visitor = unsup_type_visitor< derived_t >;
        using attr_visitor = unsup_attr_visitor< derived_t >;

        using decl_visitor::Visit;
        using stmt_visitor::Visit;
        using type_visitor::Visit;
        using attr_visitor::Visit;
    };

} // namespace vast::cg
