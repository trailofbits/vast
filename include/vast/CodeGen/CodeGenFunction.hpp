// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/DefaultGeneratorBase.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct module_context;

    //
    // function generation
    //
    struct function_generator : default_generator_base, function_scope
    {
        function_generator(
             scope_context *parent
            , const options_t &opts
            , codegen_builder &bld
            , visitor_view visitor
        )
            : default_generator_base(bld, visitor), function_scope(parent), opts(opts)
        {}

        virtual ~function_generator() = default;

        operation emit_in_scope(region_t &scope, const clang_function *decl);

        const options_t &opts;

      private:

        operation emit(const clang_function *decl);
        void declare_function_params(const clang_function *decl, vast_function fn);
    };

    //
    // function prototype generation
    //
    struct prototype_generator : default_generator_base, prototype_scope
    {
        prototype_generator(scope_context *parent, codegen_builder &bld, visitor_view visitor)
            : default_generator_base(bld, visitor), prototype_scope(parent)
        {}

        virtual ~prototype_generator() = default;

        operation emit_in_scope(region_t &scope, const clang_function *decl);

      private:

        operation emit(const clang_function *decl);
        operation lookup_or_declare(const clang_function *decl, module_context *mod);
    };

    //
    // function body generation
    //
    struct body_generator : default_generator_base, block_scope
    {
        body_generator(scope_context *parent, codegen_builder &bld, visitor_view visitor)
            : default_generator_base(bld, visitor), block_scope(parent)
        {}

        virtual ~body_generator() = default;

        void emit_in_scope(region_t &scope, const clang_function *decl, vast_function fn);

     private:

        void emit(const clang_function *decl, vast_function fn);
        void emit_epilogue(const clang_function *decl, vast_function fn);

        void deal_with_missing_return(const clang_function *decl, vast_function fn);

        bool should_final_emit_unreachable(const clang_function *decl) const;

        void emit_trap(const clang_function *decl);
        void emit_unreachable(const clang_function *decl);
        void emit_implicit_return_zero(const clang_function *decl);
        void emit_implicit_void_return(const clang_function *decl);
    };

} // namespace vast::cg
