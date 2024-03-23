// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    template< typename generator_type, typename generator_context >
    struct scope_generator : generator_context
    {
        scope_generator(visitor_view visitor, codegen_builder &bld, auto &&...args)
            : generator_context(std::forward< decltype(args) >(args)...)
            , bld(bld), visitor(visitor)
        {}

        auto & self() { return *static_cast< generator_type * >(this); }

        template< typename child_generator >
        child_generator &make_child() {
            auto &parent = self();
            parent.hook_child(std::make_unique< child_generator >(visitor, bld, &parent));
            return parent.template last_child< child_generator >();
        }

        auto do_emit(region_t &scope, auto &&...args) {
            auto _ = bld.insertion_guard();
            bld.set_insertion_point_to_end(&scope);
            return self().emit(std::forward< decltype(args) >(args)...);
        }

        codegen_builder &bld;
        visitor_view visitor;
    };

} // namespace vast::cg
