// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/Common.hpp"

namespace vast::cg {

    enum class missing_return_policy { emit_unreachable, emit_trap };

    struct codegen_policy
    {
        virtual ~codegen_policy() = default;

        virtual bool emit_strict_function_return(const clang_function *decl) const = 0;
        virtual missing_return_policy get_missing_return_policy(const clang_function *decl) const = 0;
        virtual bool skip_function_body(const clang_function *decl) const = 0;
        virtual bool skip_global_initializer(const clang_var_decl *decl) const = 0;
    };

} // namespace vast::cg
