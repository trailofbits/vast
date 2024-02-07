// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct global_emition {
        enum class kind { definition, declaration };

        bool for_vtable = false;
        bool defer = false;
        bool thunk = false;

        kind emition_kind;
    };

    static constexpr global_emition emit_definition = {
        .emition_kind = global_emition::kind::definition
    };

    static constexpr global_emition deferred_emit_definition = {
        .defer        = true,
        .emition_kind = global_emition::kind::definition
    };

    static constexpr global_emition emit_declaration = {
        .emition_kind = global_emition::kind::declaration
    };

    constexpr bool is_for_definition(global_emition emit) {
        return emit.emition_kind == global_emition::kind::definition;
    }

    constexpr bool is_for_declaration(global_emition emit) {
        return emit.emition_kind == global_emition::kind::declaration;
    }

    using vast_function = vast::hl::FuncOp;

    struct function_context : function_scope {
        virtual ~function_context() = default;
    };

    struct function_generator : function_context {
        virtual ~function_generator() = default;

        void emit(clang::FunctionDecl *decl, mangler_t &mangler);
        void emit_prologue(clang::FunctionDecl *decl, mangler_t &mangler);
        void emit_body(clang::FunctionDecl *decl);
        void emit_epilogue(clang::FunctionDecl *decl);
    };

    //
    // return potentially deferred action
    //
    std::unique_ptr< function_generator > generate_function(clang::FunctionDecl *decl, mangler_t &mangler);

} // namespace vast::cg
