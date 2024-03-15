// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/VisitorView.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    using vast_function = vast::hl::FuncOp;

    //
    // function generation
    //
    struct function_context : function_scope
    {
        using function_scope::function_scope;
        virtual ~function_context() = default;

        vast_function function;
    };

    struct function_generator : scope_generator< function_context >
    {
        using base = scope_generator< function_context >;

        function_generator(visitor_view visitor, scope_context *parent)
            : base(visitor, parent)
        {}

        virtual ~function_generator() = default;

        void emit(clang_function *decl);
    };

    //
    // function prototype generation
    //
    struct prototype_context : prototype_scope
    {
        using prototype_scope::prototype_scope;

        virtual ~prototype_context() = default;
    };

    struct prototype_generator : scope_generator< prototype_context >
    {
        using base = scope_generator< prototype_context >;

        prototype_generator(visitor_view visitor, scope_context *parent)
            : base(visitor, parent)
        {}

        virtual ~prototype_generator() = default;

        void emit(clang_function *decl);
    };

    //
    // function body generation
    //
    struct body_context : block_scope
    {
        using block_scope::block_scope;
        virtual ~body_context() = default;
    };

    struct body_generator : scope_generator< body_context >
    {
        using base = scope_generator< body_context >;

        body_generator(visitor_view visitor, scope_context *parent)
            : base(visitor, parent)
        {}

        virtual ~body_generator() = default;

        void emit(clang_function *decl);
        void emit_epilogue(clang_function *decl);
    };

    template< typename T >
    auto generate(clang_function *decl, scope_context *parent, visitor_view visitor)
        -> std::unique_ptr< T >
    {
        auto gen = std::make_unique< T >(visitor, parent);
        gen->emit(decl);
        return gen;
    }

} // namespace vast::cg
