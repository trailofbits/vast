// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/ScopeGenerator.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    // global ctor block
    struct global_context : block_scope
    {
        using block_scope::block_scope;
        virtual ~global_context() = default;
    };

    struct global_generator : scope_generator< global_generator, global_context >
    {
        using base = scope_generator< global_generator, global_context >;
        using base::base;

        virtual ~global_generator() = default;

      private:

        friend struct scope_generator< global_generator, global_context >;

        operation  emit(clang_var_decl *decl);
        mlir_value emit(clang_expr *init);
        operation lookup_or_declare(clang_var_decl *decl, module_context *mod);
    };

} // namespace vast::cg
