// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/Builder.hpp"
#include "vast/Frontend/Diagnostics.hpp"

namespace vast::cg {

    struct codegen_module {
        codegen_module(const codegen_module &) = delete;
        codegen_module &operator=(const codegen_module &) = delete;

        codegen_module(
            mcontext_t &mctx, acontext_t &actx,
            cc::diagnostics_engine &diags,
            const cc::codegen_options &cgo
        )
            : builder(mctx), actx(actx)
            , diags(diags), lang_opts(actx.getLangOpts()), codegen_opts(cgo)
            , mod( mlir::ModuleOp::create(builder.getUnknownLoc()) )
        {}

        const cc::diagnostics_engine &get_diags() const { return diags; }

        // Finalize vast code generation.
        void release();

        vast_module get_module() { return mod; }

        bool verify_module();

        // Emit any needed decls for which code generation was deferred.
        void build_deferred();

        // Helper for `buildDeferred` to apply actual codegen.
        void build_global_decl(clang::GlobalDecl &decl);

        // A queue of (optional) vtables to consider emitting.
        std::vector< const clang::CXXRecordDecl * > deferred_vtables;

        // This contains all the decls which have definitions but which are deferred
        // for emission and therefore should only be output if they are actually
        // used. If a decl is in this, then it is known to have not been referenced
        // yet.
        std::map< llvm::StringRef, clang::GlobalDecl > deferred_decls;

        // This is a list of deferred decls which we have seen that *are* actually
        // referenced. These get code generated when the module is done.
        std::vector< clang::GlobalDecl > deferred_decls_tot_emit;
        void add_deferred_decl_to_emit(clang::GlobalDecl decl) {
            deferred_decls_tot_emit.emplace_back(decl);
        }

        // After HandleTranslation finishes, differently from deferred_decls_to_emit,
        // default_methods_to_emit is only called after a set of vast passes run.
        // See add_default_methods_to_emit usage for examples.
        std::vector< clang::GlobalDecl > default_methods_to_emit;
        void add_default_methods_to_emit(clang::GlobalDecl decl) {
            default_methods_to_emit.emplace_back(decl);
        }

        void build_default_methods();

      private:

        // The builder is a helper class to create IR inside a function. The
        // builder is stateful, in particular it keeps an "insertion point": this
        // is where the next operations will be introduced.
        codegen_builder builder;

        acontext_t &actx;

        cc::diagnostics_engine &diags;

        const cc::language_options &lang_opts;
        const cc::codegen_options &codegen_opts;

        // A "module" matches a c/cpp source file: containing a list of functions.
        vast_module mod;
    };

} // namespace vast::cg
