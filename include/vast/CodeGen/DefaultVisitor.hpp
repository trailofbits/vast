// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenDeclVisitor.hpp"
#include "vast/CodeGen/CodeGenStmtVisitor.hpp"
#include "vast/CodeGen/CodeGenTypeVisitor.hpp"
#include "vast/CodeGen/CodeGenAttrVisitor.hpp"

namespace vast::cg
{
    //
    // DefaultCodeGenVisitor
    //
    // Provides default codegen for statements, declarations, types and comments.
    //
    template< typename Derived >
    struct DefaultCodeGenVisitor
        : CodeGenDeclVisitorWithAttrs< Derived >
        , CodeGenStmtVisitor< Derived >
        , CodeGenTypeVisitorWithDataLayout< Derived >
        , CodeGenAttrVisitor< Derived >
    {
        using DeclVisitor = CodeGenDeclVisitorWithAttrs< Derived >;
        using StmtVisitor = CodeGenStmtVisitor< Derived >;
        using TypeVisitor = CodeGenTypeVisitorWithDataLayout< Derived >;
        using AttrVisitor = CodeGenAttrVisitor< Derived >;

        using DeclVisitor::Visit;
        using StmtVisitor::Visit;
        using TypeVisitor::Visit;
        using AttrVisitor::Visit;
    };

} // namespace vast::cg
