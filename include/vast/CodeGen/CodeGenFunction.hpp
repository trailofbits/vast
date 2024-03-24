// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/ScopeGenerator.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct module_context;

    //
    // function generation
    //
    struct function_context : function_scope
    {
        function_context(scope_context *parent, const options_t &opts)
            : function_scope(parent), opts(opts)
        {}

        virtual ~function_context() = default;

        const options_t &opts;
    };

    struct function_generator : scope_generator< function_generator, function_context >
    {
        using base = scope_generator< function_generator, function_context >;
        using base::base;

        virtual ~function_generator() = default;

      private:

        friend struct scope_generator< function_generator, function_context >;

        operation emit(const clang_function *decl);

        void declare_function_params(const clang_function *decl, vast_function fn);
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

        operation emit(const clang_function *decl);
        operation lookup_or_declare(const clang_function *decl, module_context *mod);
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

        void emit(const clang_function *decl, vast_function fn);
        void emit_epilogue(const clang_function *decl, vast_function fn);

        void deal_with_missing_return(const clang_function *decl, vast_function fn);

        bool should_final_emit_unreachable(const clang_function *decl) const;

        insertion_guard insert_at_end(vast_function fn);

        void emit_trap(const clang_function *decl);
        void emit_unreachable(const clang_function *decl);
        void emit_implicit_return_zero(const clang_function *decl);
        void emit_implicit_void_return(const clang_function *decl);
    };

} // namespace vast::cg
