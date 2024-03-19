// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/VisitorView.hpp"

namespace vast::cg {

    template< typename scope_type >
    struct scope_generator : scope_type
    {
        explicit scope_generator(visitor_view visitor, auto &&...args)
            : scope_type(std::forward< decltype(args) >(args)...), visitor(visitor)
        {}

        virtual ~scope_generator() = default;

        template< typename child_generator >
        child_generator &make_child(auto &&...args) {
            scope_type::hook_child(std::make_unique< child_generator >(
                visitor, scope_type::symbols
            ));

            return scope_type::template last_child< child_generator >();
        }

        visitor_view visitor;
    };

} // namespace vast::cg
