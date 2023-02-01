// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/Builder.hpp"
#include "vast/CodeGen/TargetInfo.hpp"
#include "vast/CodeGen/TypesGenerator.hpp"
#include "vast/Frontend/Diagnostics.hpp"

#include "vast/Translation/CodeGen.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
namespace vast::cg {

    enum class global_emit { definition, declaration };

    struct codegen_module {
        codegen_module(const codegen_module &) = delete;
        codegen_module &operator=(const codegen_module &) = delete;

        codegen_module(
            mcontext_t &mctx, acontext_t &actx,
            cc::diagnostics_engine &diags,
            const cc::codegen_options &cgo
        )
            : actx(actx)
            , diags(diags)
            , target(actx.getTargetInfo())
            , lang_opts(actx.getLangOpts()), codegen_opts(cgo)
            , types(*this)
        {}

        const cc::diagnostics_engine &get_diags() const { return diags; }

        // Finalize vast code generation.
        void release();

        void add_replacement(string_ref name, mlir::Operation *Op);

        acontext_t &get_ast_context() { return actx; }

        bool verify_module();

        // Return the address of the given function. If ty is non-null, then this
        // function will use the specified type if it has to create it.
        // TODO: this is a bit weird as `GetAddr` given we give back a FuncOp?
        vast::hl::FuncOp get_addr_of_function(
            clang::GlobalDecl decl, mlir_type ty = nullptr,
            bool for_vtable = false, bool dontdefer = false,
            global_emit emit = global_emit::declaration
        );

        mlir::Operation *get_addr_of_function(
            clang::GlobalDecl decl,
            global_emit emit = global_emit::declaration
        );

        const target_info_t &get_target_info();

        bool should_emit_function(clang::GlobalDecl /* decl */);

        // Make sure that this type is translated.
        void update_completed_type(const clang::TagDecl *DECL_CONTEXT);

        void build_global_definition(clang::GlobalDecl decl, mlir::Operation *Op = nullptr);
        void build_global_function_definition(clang::GlobalDecl decl, mlir::Operation *op);
        void build_global_var_definition(const clang::VarDecl *decl, bool tentative = false);

        // Emit any needed decls for which code generation was deferred.
        void build_deferred();

        // Helper for `build_deferred` to apply actual codegen.
        void build_global_decl(clang::GlobalDecl &decl);

        // Emit code for a single global function or var decl. Forward declarations
        // are emitted lazily.
        void build_global(clang::GlobalDecl decl);

        void build_top_level_decl(clang::Decl *decl);

        // Determine whether the definition must be emitted; if this returns
        // false, the definition can be emitted lazily if it's used.
        bool must_be_emitted(const clang::ValueDecl *glob);

        // Determine whether the definition can be emitted eagerly, or should be
        // delayed until the end of the translation unit. This is relevant for
        // definitions whose linkage can change, e.g. implicit function instantions
        // which may later be explicitly instantiated.
        bool may_be_emitted_eagerly(const clang::ValueDecl *decl);

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
        mutable std::unique_ptr< target_info_t > target_info;

        acontext_t &actx;

        cc::diagnostics_engine &diags;

        const clang::TargetInfo &target;

        const cc::language_options &lang_opts;
        const cc::codegen_options &codegen_opts;

        // FIXME: should we use llvm::TrackingVH<mlir::Operation> here?
        using replacements_map = llvm::StringMap< mlir::Operation * >;
        replacements_map replacements;

        // Per-module type mapping from clang AST to VAST high-level types.
        types_generator types;

        // Call replaceAllUsesWith on all pairs in replacements.
        void apply_replacements();
    };

} // namespace vast::cg
