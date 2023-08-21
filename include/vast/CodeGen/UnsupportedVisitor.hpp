// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"

#include "vast/CodeGen/CodeGen.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenContext.hpp"
#include "vast/CodeGen/CodeGenVisitor.hpp"

#include <functional>

namespace vast::cg {

    template< typename Derived >
    struct UnsupportedStmtVisitor
        : clang::ConstStmtVisitor< UnsupportedStmtVisitor< Derived >, operation >
        , CodeGenVisitorLens< UnsupportedStmtVisitor< Derived >, Derived >
        , CodeGenBuilder< UnsupportedStmtVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< UnsupportedStmtVisitor< Derived >, Derived >;

        using LensType::meta_location;
        using LensType::visit;

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

    template< typename Derived >
    struct UnsupportedDeclVisitor
        : clang::ConstDeclVisitor< UnsupportedDeclVisitor< Derived >, operation >
        , CodeGenVisitorLens< UnsupportedDeclVisitor< Derived >, Derived >
        , CodeGenBuilder< UnsupportedDeclVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< UnsupportedDeclVisitor< Derived >, Derived >;

        using LensType::context;
        using LensType::meta_location;
        using LensType::visit;

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
            return make_unsupported_decl(decl);
        }
    };

    template< typename Derived >
    struct UnsupportedTypeVisitor
        : clang::TypeVisitor< UnsupportedTypeVisitor< Derived >, mlir_type >
        , CodeGenVisitorLens< UnsupportedTypeVisitor< Derived >, Derived >
        , CodeGenBuilder< UnsupportedTypeVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< UnsupportedTypeVisitor< Derived >, Derived >;
        using LensType::mcontext;

        auto make_unsupported_type(auto ty) {
            return this->template make_type< unsup::UnsupportedType >()
                .bind(&mcontext())
                .bind(ty->getTypeClassName())
                .freeze();
        }

        mlir_type Visit(clang::QualType ty) {
            return ty.isNull() ? Type() : Visit(ty.getTypePtr());
        }

        mlir_type Visit(const clang_type *ty) {
            return make_unsupported_type(ty);
        }
    };

    template< typename Derived >
    struct UnsupportedVisitor
        : UnsupportedDeclVisitor< Derived >
        , UnsupportedStmtVisitor< Derived >
        , UnsupportedTypeVisitor< Derived >
    {
        using DeclVisitor = UnsupportedDeclVisitor< Derived >;
        using StmtVisitor = UnsupportedStmtVisitor< Derived >;
        using TypeVisitor = UnsupportedTypeVisitor< Derived >;

        using DeclVisitor::Visit;
        using StmtVisitor::Visit;
        using TypeVisitor::Visit;
    };

} // namespace vast::cg
