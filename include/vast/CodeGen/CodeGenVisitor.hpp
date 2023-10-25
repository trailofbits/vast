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
    {
        using mixin          = visitor_mixin< visitor_instance< context_t, visitor_mixin, meta_generator_t > >;
        using meta_generator = meta_generator_t;
        using vast_builder   = builder_t< visitor_instance< context_t, visitor_mixin, meta_generator_t > >;

        visitor_instance(context_t &ctx, meta_generator &gen)
            : ctx(ctx), meta(gen), builder(ctx.getBodyRegion())
        {}

        using vast_builder::set_insertion_point_to_start;
        using vast_builder::set_insertion_point_to_end;
        using vast_builder::has_insertion_block;
        using vast_builder::clear_insertion_point;

        using vast_builder::make_scoped;

        using vast_builder::make_cond_builder;
        using vast_builder::make_operation;
        using vast_builder::make_region_builder;
        using vast_builder::make_stmt_expr_region;
        using vast_builder::make_type_yield_builder;
        using vast_builder::make_value_builder;
        using vast_builder::make_value_yield_region;
        using vast_builder::make_yield_true;

        using vast_builder::constant;

        using mixin::Visit;

        void set_insertion_point_to_start(region_ptr region) {
            builder.setInsertionPointToStart(&region->front());
        }

        void set_insertion_point_to_end(region_ptr region) {
            builder.setInsertionPointToEnd(&region->back());
        }

        void set_insertion_point_to_start(block_ptr block) {
            builder.setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(block_ptr block) {
            builder.setInsertionPointToEnd(block);
        }

        void clear_insertion_point() {
            builder.clearInsertionPoint();
        }

        insertion_guard make_insertion_guard() {
            return { builder };
        }

        mlir_builder& base_builder() { return builder; }

        loc_t meta_location(auto token) const {
            return meta.location(token);
        }

        context_t &ctx;
        meta_generator &meta;
        mlir_builder builder;
    };

} // namespace vast::cg
