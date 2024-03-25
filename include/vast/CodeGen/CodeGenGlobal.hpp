// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/ScopeContext.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    // global ctor block
    struct global_generator : block_scope
    {
        global_generator(scope_context *parent,  codegen_builder &bld, visitor_view visitor)
            : block_scope(parent), bld(bld), visitor(visitor)
        {}

        virtual ~global_generator() = default;

        void emit_in_scope(region_t &scope, auto decl) {
            auto _ = bld.insertion_guard();
            bld.set_insertion_point_to_end(&scope);
            emit(decl);
        }

      private:

        operation  emit(clang_var_decl *decl);
        mlir_value emit(clang_expr *init);

        operation lookup_or_declare(clang_var_decl *decl, module_context *mod);

        codegen_builder &bld;
        visitor_view visitor;
    };

} // namespace vast::cg
