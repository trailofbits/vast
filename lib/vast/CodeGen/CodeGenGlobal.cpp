// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenGlobal.hpp"

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
   operation global_generator::emit(clang_var_decl *decl) {
        auto mod = dynamic_cast< module_context* >(parent);
        VAST_CHECK(mod, "global context must be a child of a module context");
        VAST_CHECK(decl->isFileVarDecl(), "Cannot emit local var decl as global.");

        auto var = lookup_or_declare(decl, mod);

        if (decl->hasInit()) {
            defer([=] {
                auto declared = mlir::dyn_cast< hl::VarDeclOp >(var);
                auto &initializer = declared.getInitializer();
                VAST_ASSERT(initializer.empty());

                do_emit(initializer, decl->getInit());
            });
        }

        return var;
    }

    mlir_value global_generator::emit(clang_expr */* init */) {
        VAST_UNIMPLEMENTED;
    }

    // TODO deduplicate with prototype_generator::lookup_or_declare
    operation global_generator::lookup_or_declare(clang_var_decl *decl, module_context *mod) {
        if (auto symbol = visitor.symbol(decl)) {
            if (auto gv = mod->lookup_global(symbol.value())) {
                return gv;
            }
        }

        if (auto op = visitor.visit(decl)) {
            if (auto gv = mlir::dyn_cast< hl::VarDeclOp >(op)) {
                scope_context::declare(gv);
            }
            return op;
        }

        return {};
    }

} // namespace vast::cg
