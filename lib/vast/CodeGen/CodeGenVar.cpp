// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenVar.hpp"

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    void variable_generator::emit_in_scope(region_t &scope, const clang_var_decl *decl) {
        default_generator_base::emit_in_scope(scope, [&] {
            emit(decl);
        });
    }

    operation variable_generator::emit(const clang_var_decl *decl) {
        if (auto op = visitor.visit(decl)) {
            if (auto var = mlir::dyn_cast< hl::VarDeclOp >(op)) {
                scope_context::declare(var);
                if (decl->hasInit()) {
                    fill_init(decl->getInit(), var);
                }
            }

            return op;
        }

        return {};
    }

    void variable_generator::fill_init(const clang_expr *init, hl::VarDeclOp var) {
        defer([=, this] () mutable {
            auto &initializer = var.getInitializer();

            // TODO this should be expr generator (scope)?
            default_generator_base::emit_in_scope(initializer, [&] {
                bld.compose< hl::ValueYieldOp >()
                    .bind(visitor.location(init))
                    .bind_transform(visitor.visit(init), first_result)
                    .freeze();
            });
        });
    }

} // namespace vast::cg
