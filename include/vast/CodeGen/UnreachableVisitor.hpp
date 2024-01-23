// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

namespace vast::cg
{
    template< typename derived_t >
    struct unreach_stmt_visitor {
        operation Visit(const clang::Stmt *stmt) {
            VAST_FATAL("unsupported stmt: {0}", stmt->getStmtClassName());
        }
    };

    template< typename derived_t >
    struct unreach_decl_visitor {
        operation Visit(const clang::Decl *decl) {
            VAST_FATAL("unsupported decl: {0}", decl->getDeclKindName());
        }
    };

    template< typename derived_t >
    struct unreach_type_visitor {
        mlir_type Visit(clang::QualType type) {
            VAST_FATAL("unsupported type: {0}", type.getAsString());
        }

        mlir_type Visit(const clang::Type *type) {
            VAST_FATAL("unsupported type: {0}", type->getTypeClassName());
        }
    };

    template< typename derived_t >
    struct unreach_attr_visitor {
        mlir_attr Visit(const clang::Attr *attr) {
            VAST_FATAL("unsupported attr: {0}", attr->getSpelling());
        }
    };

    //
    // This is a bottom visitor, which yields an error if called
    //
    template< typename derived_t >
    struct unreach_visitor
        : unreach_decl_visitor< derived_t >
        , unreach_stmt_visitor< derived_t >
        , unreach_type_visitor< derived_t >
        , unreach_attr_visitor< derived_t >
    {
        using decl_visitor = unreach_decl_visitor< derived_t >;
        using stmt_visitor = unreach_stmt_visitor< derived_t >;
        using type_visitor = unreach_type_visitor< derived_t >;
        using attr_visitor = unreach_attr_visitor< derived_t >;

        using decl_visitor::Visit;
        using stmt_visitor::Visit;
        using type_visitor::Visit;
        using attr_visitor::Visit;
    };

} // namespace vast::cg
