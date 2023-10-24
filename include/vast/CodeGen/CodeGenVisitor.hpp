// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenScope.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenDeclVisitor.hpp"
#include "vast/CodeGen/CodeGenStmtVisitor.hpp"
#include "vast/CodeGen/CodeGenTypeVisitor.hpp"
#include "vast/CodeGen/CodeGenAttrVisitor.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

namespace vast::cg
{
    //
    // visitor_instance
    //
    // It is paramtetrized by `visitor_mixin` that implements all Visit methods.
    // This allows to cofigure Visit implementation, e.g., to provide fallback_visitor.
    //
    // `meta_generator` takes care of attaching location metadata to generated mlir primitives.
    //
    template<
        typename context_t,
        template< typename > typename visitor_mixin,
        typename meta_generator_t
    >
    struct visitor_instance
        : builder_t< visitor_instance< context_t, visitor_mixin, meta_generator_t > >
        , visitor_mixin< visitor_instance< context_t, visitor_mixin, meta_generator_t > >
        , visitor_base< context_t, meta_generator_t >
    {
        using base           = visitor_base< context_t, meta_generator_t >;
        using mixin          = visitor_mixin< visitor_instance< context_t, visitor_mixin, meta_generator_t > >;
        using meta_generator = meta_generator_t;
        using builder =
            builder_t< visitor_instance< context_t, visitor_mixin, meta_generator_t > >;

        visitor_instance(context_t &ctx, meta_generator &gen)
            : base(ctx, gen)
        {}

        using builder::set_insertion_point_to_start;
        using builder::set_insertion_point_to_end;
        using builder::has_insertion_block;
        using builder::clear_insertion_point;

        using builder::make_scoped;

        using builder::make_cond_builder;
        using builder::make_operation;
        using builder::make_region_builder;
        using builder::make_stmt_expr_region;
        using builder::make_type_yield_builder;
        using builder::make_value_builder;
        using builder::make_value_yield_region;
        using builder::make_yield_true;

        using builder::constant;

        using base::base_builder;
        using base::meta_location;
        using base::make_insertion_guard;

        using mixin::Visit;
    };

} // namespace vast::cg
