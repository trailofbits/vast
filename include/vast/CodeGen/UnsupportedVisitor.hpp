// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"

#include "vast/Translation/CodeGen.hpp"
#include "vast/Translation/CodeGenBuilder.hpp"
#include "vast/Translation/CodeGenContext.hpp"
#include "vast/Translation/CodeGenVisitor.hpp"

#include <functional>

namespace vast::cg {

    template< typename Derived >
    struct UnsupportedStmtVisitor {
        operation Visit(const clang::Stmt *stmt) {
            VAST_UNREACHABLE("unsupported stmt: {0}", stmt->getStmtClassName());
        }
    };

    template< typename Derived >
    struct UnsupportedDeclVisitor
        : clang::ConstDeclVisitor< UnsupportedDeclVisitor< Derived >, vast::Operation * >
        , CodeGenVisitorLens< UnsupportedDeclVisitor< Derived >, Derived >
        , CodeGenBuilder< UnsupportedDeclVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< UnsupportedDeclVisitor< Derived >, Derived >;

        using LensType::context;
        using LensType::meta_location;
        using LensType::visit;

        using BuilderCallBackFn =  std::function< void(Builder &, Location) >;

        auto make_body_callback(auto decl) -> std::optional< BuilderCallBackFn > {
            auto callback = [&] (auto body) {
                return [&, body] (auto &bld, auto loc) { visit(body); };
            };

            #define CALLBACK(type, body) \
            if (auto d = mlir::dyn_cast< clang::type >(decl)) { \
                return callback(d->get##body()); \
            }

            CALLBACK(StaticAssertDecl, AssertExpr)
            CALLBACK(BlockDecl, Body)
            CALLBACK(BindingDecl, Binding)
            CALLBACK(CapturedDecl, Body)
            CALLBACK(NamespaceAliasDecl, Namespace)
            CALLBACK(UsingDecl, UnderlyingDecl)
            CALLBACK(UsingShadowDecl, TargetDecl)

            #undef CALLBACK

            if (auto d = mlir::dyn_cast< clang::DeclContext >(decl)) {
                return [&] (auto &bld, auto loc) {
                    for (auto child : d->decls()) {
                        visit(child);
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
            auto callback = make_body_callback(decl);

            auto op = this->template make_operation< us::UnsupportedDecl >()
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
            return this->template make_type< us::UnsupportedType >()
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
