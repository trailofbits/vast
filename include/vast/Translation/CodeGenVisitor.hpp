// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Translation/CodeGenDeclVisitor.hpp"
#include "vast/Translation/CodeGenStmtVisitor.hpp"
#include "vast/Translation/CodeGenTypeVisitor.hpp"
#include "vast/Translation/CodeGenVisitorBase.hpp"
#include "vast/Translation/CodeGenFallBackVisitor.hpp"

namespace vast::cg
{
    //
    // DefaultCodeGenVisitorMixin
    //
    // Provides default codegen for statements, declarations, types and comments.
    //
    template< typename Derived >
    struct DefaultCodeGenVisitorMixin
        : CodeGenDeclVisitorMixin< Derived >
        , CodeGenStmtVisitorMixin< Derived >
        , CodeGenTypeVisitorWithDataLayoutMixin< Derived >
    {
        using DeclVisitor = CodeGenDeclVisitorMixin< Derived >;
        using StmtVisitor = CodeGenStmtVisitorMixin< Derived >;
        using TypeVisitor = CodeGenTypeVisitorWithDataLayoutMixin< Derived >;

        using DeclVisitor::Visit;
        using StmtVisitor::Visit;
        using TypeVisitor::Visit;
    };

    //
    // CodeGenVisitor
    //
    // It is paramtetrized by `CodeGenVisitorMixin` that implements all Visit methods.
    // This allows to cofigure Visit implementation, e.g., to provide FallBackVisitor.
    //
    // `MetaGenerator` takes care of attaching location metadata to generated mlir primitives.
    //
    template< template< typename > class CodeGenVisitorMixin, MetaGeneratorLike MetaGenerator >
    struct CodeGenVisitor
        : CodeGenVisitorMixin< CodeGenVisitor< CodeGenVisitorMixin, MetaGenerator > >
        , CodeGenVisitorBaseWithBuilder< MetaGenerator >
    {
        using BaseType          = CodeGenVisitorBaseWithBuilder< MetaGenerator >;
        using MixinType         = CodeGenVisitorMixin< CodeGenVisitor< CodeGenVisitorMixin, MetaGenerator > >;
        using MetaGeneratorType = MetaGenerator;

        CodeGenVisitor(CodeGenContext &ctx, MetaGenerator &gen)
            : BaseType(ctx, gen)
        {}

        using BaseType::set_insertion_point_to_start;
        using BaseType::has_insertion_block;
        using BaseType::clear_insertion_point;

        using MixinType::Visit;
    };

} // namespace vast::cg
