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
    template< template< typename > typename visitor_mixin >
    struct visitor_instance
        : builder_t< visitor_instance< visitor_mixin > >
        , visitor_mixin< visitor_instance< visitor_mixin > >
    {
        using mixin          = visitor_mixin< visitor_instance< visitor_mixin > >;
        using vast_builder   = builder_t< visitor_instance< visitor_mixin > >;

        visitor_instance(codegen_context &ctx, meta_generator &meta)
            : ctx(ctx), meta(meta), builder(ctx.getBodyRegion())
        {}

        using vast_builder::set_insertion_point_to_start;
        using vast_builder::set_insertion_point_to_end;
        using vast_builder::has_insertion_block;
        using vast_builder::clear_insertion_point;

        using vast_builder::insertion_guard;

        using vast_builder::make_cond_builder;
        using vast_builder::make_operation;
        using vast_builder::make_region_builder;
        using vast_builder::make_stmt_expr_region;
        using vast_builder::make_type_yield_builder;
        using vast_builder::make_value_builder;
        using vast_builder::make_value_yield_region;
        using vast_builder::make_yield_true;
        using vast_builder::make_scoped;

        using vast_builder::constant;

        using mixin::Visit;

        mlir_builder& mlir_builder() { return builder; }

        loc_t meta_location(auto token) const {
            return meta.location(token);
        }

        codegen_context &ctx;
        meta_generator &meta;
        ::vast::mlir_builder builder;
    };

} // namespace vast::cg
