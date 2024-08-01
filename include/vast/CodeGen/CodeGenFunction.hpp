// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/GeneratorBase.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct module_context;

    enum class missing_return_policy { emit_unreachable, emit_trap };

    mlir_visibility get_function_visibility(const clang_function *decl, linkage_kind linkage);
    vast_function set_visibility(const clang_function *decl, vast_function fn);
    vast_function set_linkage_and_visibility(vast_function fn, std::optional< core::GlobalLinkageKind > linkage);

    //
    // function generation
    //
    struct function_generator : generator_base
    {
        using scope_type = function_scope;

        using generator_base::generator_base;

        operation emit(const clang_function *decl);
        void declare_function_params(const clang_function *decl, vast_function fn);

        void emit_body(const clang_function *decl, vast_function prototype);
        void emit_epilogue(const clang_function *decl, vast_function prototype);

        void emit_labels(const clang_function *decl, vast_function prototype);

        void deal_with_missing_return(const clang_function *decl, vast_function fn);
        bool should_final_emit_unreachable(const clang_function *decl) const;

        void emit_trap(const clang_function *decl);
        void emit_unreachable(const clang_function *decl);
        void emit_implicit_return_zero(const clang_function *decl);
        void emit_implicit_void_return(const clang_function *decl);

        bool emit_strict_function_return;
        missing_return_policy missing_return_policy;
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
