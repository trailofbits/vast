// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Translation/CodeGenDeclVisitor.hpp"
#include "vast/Translation/CodeGenStmtVisitor.hpp"
#include "vast/Translation/CodeGenTypeVisitor.hpp"

namespace vast::cg
{
    //
    // DefaultCodeGenVisitor
    //
    // Provides default codegen for statements, declarations, types and comments.
    //
    template< typename Derived >
    struct DefaultCodeGenVisitor
        : CodeGenDeclVisitor< Derived >
        , CodeGenStmtVisitor< Derived >
        , CodeGenTypeVisitorWithDataLayout< Derived >
    {
        using DeclVisitor = CodeGenDeclVisitor< Derived >;
        using StmtVisitor = CodeGenStmtVisitor< Derived >;
        using TypeVisitor = CodeGenTypeVisitorWithDataLayout< Derived >;

        using DeclVisitor::Visit;
        using StmtVisitor::Visit;
        using TypeVisitor::Visit;
    };

} // namespace vast::cg
