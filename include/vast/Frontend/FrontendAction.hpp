// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/FrontendAction.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Options.hpp"
#include "vast/Frontend/CompilerInstance.hpp"

namespace vast::cc {

    using frontend_action = clang::ASTFrontendAction;
    using plugin_ast_action = clang::PluginASTAction;

    static inline action_options options(compiler_instance &ci) {
        return {
            .headers = ci.getHeaderSearchOpts(),
            .codegen = ci.getCodeGenOpts(),
            .target  = ci.getTargetOpts(),
            .lang    = ci.getLangOpts(),
            .front   = ci.getFrontendOpts(),
            .diags   = ci.getDiagnostics(),
            .vfs     = ci.getVirtualFileSystem()
        };
    }

} // namespace vast::cc
