// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/FrontendAction.h>
VAST_UNRELAX_WARNINGS

namespace vast::cc {

    using frontend_action = clang::ASTFrontendAction;
    using plugin_ast_action = clang::PluginASTAction;

    enum class output_type {
        emit_assembly,
        emit_high_level,
        emit_cir,
        emit_llvm,
        emit_obj,
        none
    };

} // namespace vast::cc
