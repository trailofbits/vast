// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/Common.hpp"

namespace vast::cg {

    enum class missing_return_policy { emit_unreachable, emit_trap };

    struct policy_base
    {
        virtual ~policy_base() = default;

        virtual bool emit_strict_function_return([[maybe_unused]] const clang_function *decl
        ) const = 0;

        virtual missing_return_policy
        missing_return_policy([[maybe_unused]] const clang_function *decl) const = 0;

        virtual bool skip_function_body([[maybe_unused]] const clang_function *decl) const = 0;
    };

} // namespace vast::cg
