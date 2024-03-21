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

        defer([=] {
            //auto &bg = make_child< body_generator >();
            // TODO pass prototype to body generator
            // bg.emit(decl);
        });

        return prototype;
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
