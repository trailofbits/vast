// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg {

    struct visitor_view
    {
        using visit_decl_func = std::function< void(clang::Decl *) >;
        using visit_stmt_func = std::function< void(clang::Stmt *) >;
        using visit_type_func = std::function< void(clang::Type *) >;
        using visit_attr_func = std::function< void(clang::Attr *) >;

        template< typename visitor_t >
        explicit visitor_view(visitor_t &visitor)
            : visit_decl([&visitor](clang::Decl *decl) { visitor.Visit(decl); })
            , visit_stmt([&visitor](clang::Stmt *stmt) { visitor.Visit(stmt); })
            , visit_type([&visitor](clang::Type *type) { visitor.Visit(type); })
            , visit_attr([&visitor](clang::Attr *attr) { visitor.Visit(attr); })
        {}

        void Visit(clang::Decl *decl) { visit_decl(decl); }
        void Visit(clang::Stmt *stmt) { visit_stmt(stmt); }
        void Visit(clang::Type *type) { visit_type(type); }
        void Visit(clang::Attr *attr) { visit_attr(attr); }

      private:
        visit_decl_func visit_decl;
        visit_stmt_func visit_stmt;
        visit_type_func visit_type;
        visit_attr_func visit_attr;
    };

} // namespace vast::cg
