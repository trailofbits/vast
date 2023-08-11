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

namespace vast::cg {

    template< typename Derived >
    struct UnsupportedStmtVisitor {
        operation Visit(const clang::Stmt *stmt) {
            VAST_UNREACHABLE("unsupported stmt: {0}", stmt->getStmtClassName());
        }
    };

    template< typename Derived >
    struct UnsupportedDeclVisitor {
        operation Visit(const clang::Decl *decl) {
            // TODO
            VAST_UNREACHABLE("unsupported decl: {0}", decl->getDeclKindName());
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
        using LensType::visit;

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
