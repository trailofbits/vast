// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"
#include "vast/CodeGen/GeneratorBase.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct module_context;

    //
    // function generation
    //
    struct function_generator : generator_base
    {
        using scope_type = function_scope;

        function_generator(codegen_builder &bld, scoped_visitor_view visitor, const options_t &opts)
            : generator_base(bld, visitor), opts(opts)
        {}

        using generator_base::generator_base;
        virtual ~function_generator() = default;

        operation emit(const clang_function *decl);
        void declare_function_params(const clang_function *decl, vast_function fn);

        void emit_body(const clang_function *decl, vast_function prototype);
        void emit_epilogue(const clang_function *decl, vast_function prototype);

        void deal_with_missing_return(const clang_function *decl, vast_function fn);
        bool should_final_emit_unreachable(const clang_function *decl) const;

        void emit_trap(const clang_function *decl);
        void emit_unreachable(const clang_function *decl);
        void emit_implicit_return_zero(const clang_function *decl);
        void emit_implicit_void_return(const clang_function *decl);

        const options_t &opts;
    };

    //
    // function prototype generation
    //
    struct prototype_generator : generator_base
    {
        using scope_type = prototype_scope;

        using generator_base::generator_base;
        virtual ~prototype_generator() = default;

        operation emit(const clang_function *decl);
    };

} // namespace vast::cg
