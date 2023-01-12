// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
VAST_UNRELAX_WARNINGS

namespace vast::cc
{
    using frontend_action_ptr   = std::unique_ptr< clang::FrontendAction >;
    using compiler_instance     = clang::CompilerInstance;
} // namespace vast::cc
