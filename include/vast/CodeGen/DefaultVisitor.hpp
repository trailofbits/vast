// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenDeclVisitor.hpp"
#include "vast/CodeGen/CodeGenStmtVisitor.hpp"
#include "vast/CodeGen/CodeGenTypeVisitor.hpp"
#include "vast/CodeGen/CodeGenAttrVisitor.hpp"

namespace vast::cg
{
    //
    // default_visitor
    //
    // Provides default codegen for statements, declarations, types and comments.
    //
    template< typename derived_t >
    struct default_visitor
        : decl_visitor_with_attrs< derived_t >
        , default_stmt_visitor< derived_t >
        , type_visitor_with_dl< derived_t >
        , default_attr_visitor< derived_t >
    {
        using decl_visitor = decl_visitor_with_attrs< derived_t >;
        using stmt_visitor = default_stmt_visitor< derived_t >;
        using type_visitor = type_visitor_with_dl< derived_t >;
        using attr_visitor = default_attr_visitor< derived_t >;

        using decl_visitor::Visit;
        using stmt_visitor::Visit;
        using type_visitor::Visit;
        using attr_visitor::Visit;
    };

} // namespace vast::cg
