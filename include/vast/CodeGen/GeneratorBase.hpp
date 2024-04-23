// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    struct generator_base
    {
        generator_base(codegen_builder &bld, scoped_visitor_view visitor)
            : bld(bld), visitor(visitor)
        {}

        scope_context &scope() { return visitor.scope; }

        void defer(scope_context::deferred_task &&task) {
            visitor.scope.defer(std::move(task));
        }

        codegen_builder &bld;
        scoped_visitor_view visitor;
    };

    template< typename generator, typename scope, typename... args_t >
    generator mk_scoped_generator(
        scope_context &parent, codegen_builder &bld, visitor_view visitor, args_t &&... args
    ) {
        auto &new_scope = parent.mk_child< scope >();
        return generator(
            bld, scoped_visitor_view(*visitor.raw(), new_scope),
            std::forward< args_t >(args)...
        );
    }

    template< typename generator, typename... args_t >
    generator mk_scoped_generator(
        scope_context &scope, codegen_builder &bld, visitor_view visitor, args_t &&... args
    ) {
        using scope_type = typename generator::scope_type;
        return mk_scoped_generator< generator, scope_type >(
            scope, bld, visitor,
            std::forward< args_t >(args)...
        );
    }

    template< typename generator, typename... args_t >
    generator mk_scoped_generator(generator_base &parent, args_t &&... args) {
        return mk_scoped_generator< generator >(
            parent.scope(), parent.bld, parent.visitor,
            std::forward< args_t >(args)...
        );
    }

} // namespace vast::cg
