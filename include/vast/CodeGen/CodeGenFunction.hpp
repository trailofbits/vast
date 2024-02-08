// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/Mangler.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    using vast_function = vast::hl::FuncOp;

    //
    // function generation
    //
    struct function_context : function_scope {
        virtual ~function_context() = default;
    };

    struct function_generator : function_context {
        virtual ~function_generator() = default;

        void emit(clang::FunctionDecl *decl, mangler_t &mangler);
    };

    std::unique_ptr< function_generator > generate_function(
        clang::FunctionDecl *decl, mangler_t &mangler
    );

    //
    // function prototype generation
    //
    struct prototype_context : prototype_scope {
        virtual ~prototype_context() = default;
    };

    struct prototype_generator : prototype_context {
        virtual ~prototype_generator() = default;

        void emit(clang::FunctionDecl *decl, mangler_t &mangler);
    };

    std::unique_ptr< prototype_generator > generate_prototype(
        clang::FunctionDecl *decl, mangler_t &mangler
    );

    //
    // function body generation
    //
    struct body_context : block_scope {
        virtual ~body_context() = default;
    };

    struct body_generator : body_context {
        virtual ~body_generator() = default;

        void emit(clang::FunctionDecl *decl);
        void emit_epilogue(clang::FunctionDecl *decl);
    };

    std::unique_ptr< body_generator > generate_body(
        clang::FunctionDecl *decl, emition_kind emition
    );

} // namespace vast::cg
