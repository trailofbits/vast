// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/ScopeGenerator.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct module_context;

    //
    // function generation
    //
    struct function_context : function_scope
    {
        using function_scope::function_scope;
        virtual ~function_context() = default;
    };

    struct function_generator : scope_generator< function_generator, function_context >
    {
        using base = scope_generator< function_generator, function_context >;
        using base::base;

        virtual ~function_generator() = default;

      private:

        friend struct scope_generator< function_generator, function_context >;

        operation emit(clang_function *decl);

        void declare_function_params(vast_function fn, clang_function *decl);
    };

    //
    // function prototype generation
    //
    struct prototype_context : prototype_scope
    {
        using prototype_scope::prototype_scope;
        virtual ~prototype_context() = default;
    };

    struct prototype_generator : scope_generator< prototype_generator, prototype_context >
    {
        using base = scope_generator< prototype_generator, prototype_context >;
        using base::base;

        virtual ~prototype_generator() = default;

      private:

        friend struct scope_generator< prototype_generator, prototype_context >;

        operation emit(clang_function *decl);
        operation lookup_or_declare(clang_function *decl, module_context *mod);
    };

    //
    // function body generation
    //
    struct body_context : block_scope
    {
        using block_scope::block_scope;
        virtual ~body_context() = default;
    };

    struct body_generator : scope_generator< body_generator, body_context >
    {
        using base = scope_generator< body_generator, body_context >;
        using base::base;

        virtual ~body_generator() = default;

     private:

        friend struct scope_generator< body_generator, body_context >;

        void emit(clang_function *decl);
        void emit_epilogue(clang_function *decl);
    };

} // namespace vast::cg
