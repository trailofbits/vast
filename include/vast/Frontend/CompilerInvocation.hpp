// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/CompilerInvocation.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Common.hpp"
#include "vast/Frontend/Diagnostics.hpp"
#include "vast/Frontend/CompilerInstance.hpp"
#include "vast/Frontend/Options.hpp"

namespace vast::cc
{
    using clang_invocation = clang::CompilerInvocation;

    struct compiler_invocation {
        static bool create_from_args(clang_invocation &inv, diagnostics_engine &engine, argv_t argv, arg_t argv0) {
            return clang_invocation::CreateFromArgs(inv, argv, engine, argv0);
        }
    };

} // namespace vast::cc
