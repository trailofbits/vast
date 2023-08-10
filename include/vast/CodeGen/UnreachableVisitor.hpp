// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

namespace vast::cg
{
    template< typename Derived >
    struct UnreachableStmtVisitor {
        operation Visit(const clang::Stmt *stmt) {
            VAST_UNREACHABLE("unsupported stmt: {0}", stmt->getStmtClassName());
        }
    };

    template< typename Derived >
    struct UnreachableDeclVisitor {
        operation Visit(const clang::Decl *decl) {
            VAST_UNREACHABLE("unsupported decl: {0}", decl->getDeclKindName());
        }
    };

    template< typename Derived >
    struct UnreachableTypeVisitor {
        mlir_type Visit(clang::QualType type) {
            VAST_UNREACHABLE("unsupported type: {0}", type.getAsString());
        }

        mlir_type Visit(const clang::Type *type) {
            VAST_UNREACHABLE("unsupported type: {0}", type->getTypeClassName());
        }
    };

    template< typename Derived >
    struct UnreachableVisitor
        : UnreachableDeclVisitor< Derived >
        , UnreachableStmtVisitor< Derived >
        , UnreachableTypeVisitor< Derived >
    {
        using DeclVisitor = UnreachableDeclVisitor< Derived >;
        using StmtVisitor = UnreachableStmtVisitor< Derived >;
        using TypeVisitor = UnreachableTypeVisitor< Derived >;

        using DeclVisitor::Visit;
        using StmtVisitor::Visit;
        using TypeVisitor::Visit;
    };

} // namespace vast::cg
