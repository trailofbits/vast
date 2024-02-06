// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg
{
    void function_generator::emit(clang::FunctionDecl *decl) {
        emit_prologue(decl);
        defer([this, decl] {
            emit_body(decl);
            emit_epilogue(decl);
        });
    }

    void function_generator::emit_prologue(clang::FunctionDecl *decl) {
        VAST_REPORT("emit prologue {0}", decl->getName());
    }

    void function_generator::emit_body(clang::FunctionDecl *decl) {
        VAST_REPORT("emit body {0}", decl->getName());
    }

    void function_generator::emit_epilogue(clang::FunctionDecl *decl) {
        VAST_REPORT("emit epilogue {0}", decl->getName());
    }

    function_context generate_function(
        clang::FunctionDecl *decl, scope_context &parent
    ) {
        function_generator gen(parent);
        gen.emit(decl);
        return std::move(gen); // return as a deferred scope
    }

} // namespace vast::cg
