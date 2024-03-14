// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/AttrVisitor.h>
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS


#include "vast/Util/Common.hpp"

namespace vast::cg {

    template< typename derived_t >
    using decl_visitor_base = clang::ConstDeclVisitor< derived_t, operation >;

    template< typename derived_t >
    using stmt_visitor_base = clang::ConstStmtVisitor< derived_t, operation >;

    template< typename derived_t >
    using type_visitor_base = clang::TypeVisitor< derived_t, mlir_type >;

    template< typename derived_t >
    using attr_visitor_base = clang::ConstAttrVisitor< derived_t, mlir_attr >;

    struct visitor_base
    {
        virtual ~visitor_base() = default;

        virtual void Visit(clang::Decl *decl) = 0;
        virtual void Visit(clang::Stmt *stmt) = 0;
        virtual void Visit(clang::Type *type) = 0;
        virtual void Visit(clang::Attr *attr) = 0;
    };

} // namespace vast::cg
