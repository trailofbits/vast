// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Translation/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Verifier.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/Error.hpp"

namespace vast::cg
{
    defer_handle_of_top_level_decl::defer_handle_of_top_level_decl(
        codegen_driver &codegen, bool emit_deferred
    )
        : codegen(codegen), emit_deferred(emit_deferred)
    {
        ++codegen.deffered_top_level_decls;
    }

    defer_handle_of_top_level_decl::~defer_handle_of_top_level_decl() {
        unsigned level = --codegen.deffered_top_level_decls;
        if (level == 0 && emit_deferred) {
            codegen.build_deferred_decls();
        }
    }

    bool codegen_driver::verify_module() const {
        return mlir::verify(mod).succeeded();
    }

    void codegen_driver::build_deferred_decls() {
        if (deferred_inline_member_func_defs.empty())
            return;

        // Emit any deferred inline method definitions. Note that more deferred
        // methods may be added during this loop, since ASTConsumer callbacks can be
        // invoked if AST inspection results in declarations being added.
        auto deferred = deferred_inline_member_func_defs;
        deferred_inline_member_func_defs.clear();

        defer_handle_of_top_level_decl defer(*this);
        for (auto decl : deferred) {
            handle_top_level_decl(decl);
        }

        // Recurse to handle additional deferred inline method definitions.
        build_deferred_decls();
    }

    void codegen_driver::build_default_methods() {
        throw cg::unimplemented("build_default_methods");
    }

    void codegen_driver::handle_translation_unit(acontext_t &/* acontext */) { /* noop */ }

    bool codegen_driver::handle_top_level_decl(clang::DeclGroupRef /* decls */) {
        throw cg::unimplemented("handle_top_level_decl");
    }

    bool codegen_driver::handle_top_level_decl(clang::Decl */* decl */) {
        throw cg::unimplemented("handle_top_level_decl");
        (void)mctx;
    }

} // namespace vast::cg
