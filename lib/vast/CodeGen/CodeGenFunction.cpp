// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg
{
    void function_generator::emit(clang::FunctionDecl *decl, mangler_t &mangler) {
        emit_prologue(decl, mangler);
        defer([this, decl] {
            emit_body(decl);
            emit_epilogue(decl);
        });
    }

    void function_generator::emit_prologue(clang::FunctionDecl *decl, mangler_t &mangler) {
        VAST_REPORT("emit prologue {0}", decl->getName());
    }

    void function_generator::emit_body(clang::FunctionDecl *decl) {
        VAST_REPORT("emit body {0}", decl->getName());
    }

    void function_generator::emit_epilogue(clang::FunctionDecl *decl) {
        VAST_REPORT("emit epilogue {0}", decl->getName());
    }

    std::unique_ptr< function_generator > generate_function(
        clang::FunctionDecl *decl, mangler_t &mangler
    ) {
        auto gen = std::make_unique< function_generator >();
        gen->emit(decl, mangler);
        return gen;
    }

} // namespace vast::cg
