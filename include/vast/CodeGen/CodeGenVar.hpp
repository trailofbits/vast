// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct variable_generator : default_generator_base, inherited_scope
    {
        variable_generator(scope_context *parent,  codegen_builder &bld, visitor_view visitor)
            : default_generator_base(bld, visitor), inherited_scope(parent)
        {}

        virtual ~variable_generator() = default;

        void emit_in_scope(region_t &scope, const clang_var_decl *decl);
        operation emit(const clang_var_decl *decl);

      private:
        void fill_init(const clang_expr *init, hl::VarDeclOp var);
    };

    void mk_var(auto &parent, const clang_var_decl *decl) {
        auto &vg = parent.template mk_child< variable_generator >(parent.bld, parent.visitor);
        vg.emit(decl);
    }

    void mk_var_in_scope(auto &parent, region_t &region, const clang_var_decl *decl) {
        auto &vg = parent.template mk_child< variable_generator >(parent.bld, parent.visitor);
        vg.emit_in_scope(region, decl);
    }


} // namespace vast::cg
