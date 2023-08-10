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
    struct UnsupportedTypeVisitor {
        mlir_type Visit(clang::QualType type) {
            // TODO
            VAST_UNREACHABLE("unsupported type: {0}", type.getAsString());
        }

        mlir_type Visit(const clang::Type *type) {
            // TODO
            VAST_UNREACHABLE("unsupported type: {0}", type->getTypeClassName());
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
