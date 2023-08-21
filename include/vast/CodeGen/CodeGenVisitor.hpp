// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenScope.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenDeclVisitor.hpp"
#include "vast/CodeGen/CodeGenStmtVisitor.hpp"
#include "vast/CodeGen/CodeGenTypeVisitor.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

namespace vast::cg
{
    //
    // CodeGenVisitor
    //
    // It is paramtetrized by `CodeGenVisitor` that implements all Visit methods.
    // This allows to cofigure Visit implementation, e.g., to provide FallBackVisitor.
    //
    // `MetaGenerator` takes care of attaching location metadata to generated mlir primitives.
    //
    template<
        typename CGContext,
        template< typename > typename VisitorMixin,
        MetaGeneratorLike MetaGenerator
    >
    struct CodeGenVisitor
        : VisitorMixin< CodeGenVisitor< CGContext, VisitorMixin, MetaGenerator > >
        , CodeGenVisitorBase< CGContext, MetaGenerator >
    {
        using BaseType          = CodeGenVisitorBase< CGContext, MetaGenerator >;
        using Mixin             = VisitorMixin< CodeGenVisitor< CGContext, VisitorMixin, MetaGenerator > >;
        using MetaGeneratorType = MetaGenerator;

        CodeGenVisitor(CGContext &ctx, MetaGenerator &gen)
            : BaseType(ctx, gen)
        {}

        using BaseType::set_insertion_point_to_start;
        using BaseType::has_insertion_block;
        using BaseType::clear_insertion_point;

        using Mixin::Visit;
    };

} // namespace vast::cg
