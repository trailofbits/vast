// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

namespace vast::hl
{
    template< typename Derived >
    struct DefaultFallBackStmtVisitorMixin {
        Operation* Visit(const clang::Stmt *stmt) {
            stmt->dump();
            VAST_UNREACHABLE("unsupported stmt: {0}", stmt->getStmtClassName());
        }
    };

    template< typename Derived >
    struct DefaultFallBackDeclVisitorMixin {
        Operation* Visit(const clang::Decl *decl) {
            VAST_UNREACHABLE("unsupported decl: {0}", decl->getDeclKindName());
        }
    };

    template< typename Derived >
    struct DefaultFallBackTypeVisitorMixin {
        Type Visit(clang::QualType type) {
            VAST_UNREACHABLE("unsupported type: {0}", type.getAsString());
        }

        Type Visit(const clang::Type *type) {
            VAST_UNREACHABLE("unsupported type: {0}", type->getTypeClassName());
        }
    };

    template< typename Derived >
    struct DefaultFallBackVisitorMixin
        : DefaultFallBackDeclVisitorMixin< Derived >
        , DefaultFallBackStmtVisitorMixin< Derived >
        , DefaultFallBackTypeVisitorMixin< Derived >
    {
        using DeclVisitor = DefaultFallBackDeclVisitorMixin< Derived >;
        using StmtVisitor = DefaultFallBackStmtVisitorMixin< Derived >;
        using TypeVisitor = DefaultFallBackTypeVisitorMixin< Derived >;

        using DeclVisitor::Visit;
        using StmtVisitor::Visit;
        using TypeVisitor::Visit;
    };

    //
    // CodeGenFallBackVisitorMixin
    //
    // Allows to specify fallback in case that `DefaultMixin::Visit` is unsuccessful.
    //
    // FallBackMixin needs to implement fallback implementation of `Visit` function.
    //
    template< typename Derived
        , template< typename > typename DefaultMixin
        , template< typename > typename FallBackMixin = DefaultFallBackVisitorMixin
    >
    struct CodeGenFallBackVisitorMixin
        : DefaultMixin< Derived >
        , FallBackMixin< Derived >
    {
        using DefaultVisitorMixin  = DefaultMixin< Derived >;
        using FallBackVisitorMixin = FallBackMixin< Derived >;

        Operation* Visit(const clang::Stmt *stmt) { return VisitWithFallBack(stmt); }
        Operation* Visit(const clang::Decl *decl) { return VisitWithFallBack(decl); }
        Type       Visit(const clang::Type *type) { return VisitWithFallBack(type); }
        Type       Visit(clang::QualType    type) { return VisitWithFallBack(type); }
      private:
        auto VisitWithFallBack(auto token) {
            if (auto result = DefaultVisitorMixin::Visit(token))
                return result;
            return FallBackVisitorMixin::Visit(token);
        }
    };

} // namespace vast::hl
