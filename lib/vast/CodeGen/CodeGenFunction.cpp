// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenFunction.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenModule.hpp"

namespace vast::cg
{
   void function_generator::emit(clang_function *decl) {
        auto ctx = dynamic_cast< module_context* >(parent);
        VAST_CHECK(ctx, "function generator must be a child of a module context");
        generate_child< prototype_generator >(decl);
        defer([=] { generate_child< body_generator >(decl); });
    }

    void prototype_generator::emit(clang_function *decl) {
        auto ctx = dynamic_cast< function_context* >(parent);
        VAST_CHECK(ctx, "prototype generator must be a child of a function context");

        auto mod = dynamic_cast< module_context* >(ctx->parent);
        VAST_CHECK(mod, "function context must be a child of a module context");

        if (auto proto = get_global_value(mod, clang_global(decl))) {
            return;
        }

        // emit_function_type(decl->getFunctionType(), decl->isVariadic());
        // get_or_create_vast_function(mangled_name, fty, decl, emit);
        // auto fn = create_vast_function(meta_location(decl), mangled_name, fty, function_decl);
    }

    // mlir_type prototype_generator::emit_function_type(
    //     const clang_function_type *fty, bool is_variadic
    // ) {
    //     VAST_UNIMPLEMENTED;
    // }

    void body_generator::emit(clang_function *decl) {
        emit_epilogue(decl);
    }

    void body_generator::emit_epilogue(clang_function *decl) {}

} // namespace vast::cg
