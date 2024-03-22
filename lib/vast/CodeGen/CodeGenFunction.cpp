// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenFunction.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/Util.hpp"

#include "vast/Util/Maybe.hpp"

#include "vast/Dialect/Core/Linkage.hpp"

namespace vast::cg
{
    //
    // function generation
    //

   operation function_generator::emit(clang_function *decl) {
        auto ctx = dynamic_cast< module_context* >(parent);
        VAST_CHECK(ctx, "function context must be a child of a module context");

        auto &pg = make_child< prototype_generator >();
        auto prototype = pg.do_emit(ctx->mod->getBodyRegion(), decl);

        if (auto fn = mlir::dyn_cast< vast_function >(prototype)) {
            if (decl->hasBody()) {
                declare_function_params(fn, decl);

        defer([=] {
                    if (auto fn = mlir::dyn_cast< vast_function >(prototype)) {
                        auto &bg = make_child< body_generator >();
                        bg.do_emit(fn.getBody(), decl);
                    } else {
                        VAST_REPORT("can not emit function body for unknown prototype");
                    }
                });
            }
        }

        return prototype;
    }

    void function_generator::declare_function_params(vast_function fn, clang_function *decl) {
        auto *entry_block = fn.addEntryBlock();
        auto params = llvm::zip(decl->parameters(), entry_block->getArguments());
        for (const auto &[param, earg] : params) {
            // TODO set alignment

            earg.setLoc(visitor.location(param));
            if (auto name = visitor.symbol(param)) {
                // TODO set name
                scope_context::declare(name.value(), earg);
            }
        }
    }

    //
    // function prototype generation
    //

    operation prototype_generator::lookup_or_declare(clang_function *decl, module_context *mod) {
        if (auto symbol = visitor.symbol(decl)) {
            if (auto fn = mod->lookup_global(symbol.value())) {
                return fn;
            }
        }

        if (auto op = visitor.visit_prototype(decl)) {
            if (auto fn = mlir::dyn_cast< vast_function >(op)) {
                scope_context::declare(fn);
            }

            return op;
        }

        return {};
    }

    operation prototype_generator::emit(clang_function *decl) {
        auto ctx = dynamic_cast< function_context* >(parent);
        VAST_CHECK(ctx, "prototype generator must be a child of a function context");

        auto mod = dynamic_cast< module_context* >(ctx->parent);
        VAST_CHECK(mod, "function context must be a child of a module context");

        return lookup_or_declare(decl, mod);
    }

    //
    // function body generation
    //

    void body_generator::emit(clang_function *decl) {
        emit_epilogue(decl);
    }

    void body_generator::emit_epilogue(clang_function *decl) {}

} // namespace vast::cg
