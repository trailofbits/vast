// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    struct default_generator_base
    {
        default_generator_base(codegen_builder &bld, visitor_view visitor)
            : bld(bld), visitor(visitor)
        {}

        template< typename emit_type_t >
        decltype(auto) emit_in_scope(region_t &scope, emit_type_t &&emit) {
            auto _ = bld.scoped_insertion_at_end(&scope);
            return emit();
        }

        codegen_builder &bld;
        visitor_view visitor;
    };
} // namespace vast::cg
