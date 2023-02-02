// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Translation/CodeGen.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::cg
{
    struct codegen_driver_options {
        bool verbose_diagnostics = true;
        // forwarded options form clang codegen
        unsigned int coverage_mapping = false;
        unsigned int keep_static_consts = false;
    };

    struct codegen_driver;

    // Use this when emitting decls to block re-entrant decl emission. It will
    // emit all deferred decls on scope exit. Set emit_deferred to false if decl
    // emission must be deferred longer, like at the end of a tag definition.
    struct defer_handle_of_top_level_decl {
        codegen_driver &codegen;
        bool emit_deferred;

        explicit defer_handle_of_top_level_decl(
            codegen_driver &codegen, bool emit_deferred = true
        );

        ~defer_handle_of_top_level_decl();
    };

    // tags for globals emition
    enum class global_emit { definition, declaration };

    // This is a layer that provides interface between
    // clang codegen and vast codegen
    struct codegen_driver {

        explicit codegen_driver(
            acontext_t &actx, mcontext_t &mctx, codegen_driver_options opts
        )
            : actx(actx)
            , mctx(mctx)
            , options(opts)
            , codegen(&actx, &mctx)
        {}

        ~codegen_driver() {
            assert(deferred_inline_member_func_defs.empty());
        }

        codegen_driver(const codegen_driver &) = delete;
        codegen_driver(codegen_driver &&) = delete;

        codegen_driver& operator=(const codegen_driver &) = delete;
        codegen_driver& operator=(codegen_driver &&) = delete;

        bool verify_module() const;

        void handle_translation_unit(acontext_t &acontext);
        void handle_top_level_decl(clang::DeclGroupRef decls);
        void handle_top_level_decl(clang::Decl *decl);

        // vast_module take_module() { return std::move(mod); }

        void finalize();
        owning_module_ref freeze();

    private:
        // Return the address of the given function. If ty is non-null, then this
        // function will use the specified type if it has to create it.
        // TODO: this is a bit weird as `get_addr` given we give back a FuncOp?
        vast::hl::FuncOp get_addr_of_function(
            clang::GlobalDecl decl, mlir_type ty = nullptr,
            bool for_vtable = false, bool dontdefer = false,
            global_emit emit = global_emit::declaration
        );

        mlir::Operation *get_addr_of_function(
            clang::GlobalDecl decl,
            global_emit emit = global_emit::declaration
        );

        bool should_emit_function(clang::GlobalDecl decl);

        void build_global_definition(clang::GlobalDecl decl, mlir::Operation *Op = nullptr);
        void build_global_function_definition(clang::GlobalDecl decl, mlir::Operation *op);
        void build_global_var_definition(const clang::VarDecl *decl, bool tentative = false);

        // Emit any needed decls for which code generation was deferred.
        void build_deferred();

        // Helper for `build_deferred` to apply actual codegen.
        void build_global_decl(clang::GlobalDecl &decl);
        // Emit code for a single global function or var decl.
        // Forward declarations are emitted lazily.
        void build_global(clang::GlobalDecl decl);

        // Determine whether the definition must be emitted; if this returns
        // false, the definition can be emitted lazily if it's used.
        bool must_be_emitted(const clang::ValueDecl *glob);

        // Determine whether the definition can be emitted eagerly, or should be
        // delayed until the end of the translation unit. This is relevant for
        // definitions whose linkage can change, e.g. implicit function
        // instantions which may later be explicitly instantiated.
        bool may_be_emitted_eagerly(const clang::ValueDecl *decl);

        void build_deferred_decls();
        void build_default_methods();

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

        // FIXME: should we use llvm::TrackingVH<mlir::Operation> here?
        using replacements_map = llvm::StringMap< mlir::Operation * >;
        replacements_map replacements;

        void add_replacement(string_ref name, mlir::Operation *Op);

        // Call replaceAllUsesWith on all pairs in replacements.
        void apply_replacements();

        inline auto lang() const { return actx.getLangOpts(); }

        acontext_t &actx;
        mcontext_t &mctx;

        codegen_driver_options options;

        unsigned deffered_top_level_decls = 0;

        friend struct defer_handle_of_top_level_decl;

        llvm::SmallVector< clang::FunctionDecl *, 8 > deferred_inline_member_func_defs;

        // FIXME: make configurable
        hl::CodeGenWithMetaIDs codegen;
    };

} // namespace vast::cg
