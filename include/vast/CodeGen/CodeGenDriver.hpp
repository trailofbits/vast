// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/CodeGen/CodeGen.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::cg
{
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

    meta_generator_ptr make_meta_generator(codegen_context &cgctx, const cc::vast_args &vargs);

    // This is a layer that provides interface between
    // clang codegen and vast codegen

    struct codegen_driver {

        explicit codegen_driver(
            codegen_context &cgctx, cc::action_options &opts, const cc::vast_args &vargs
        )
            : cgctx(cgctx)
            , opts(opts)
            , vargs(vargs)
            , meta(make_meta_generator(cgctx, vargs))
            , codegen(cgctx, *meta)
        {}

        ~codegen_driver() {
            VAST_ASSERT(deferred_inline_member_func_defs.empty());
        }

        codegen_driver(const codegen_driver &) = delete;
        codegen_driver(codegen_driver &&) = delete;

        codegen_driver& operator=(const codegen_driver &) = delete;
        codegen_driver& operator=(codegen_driver &&) = delete;

        bool verify_module() const;

        void handle_top_level_decl(clang::DeclGroupRef decls);
        void handle_top_level_decl(clang::Decl *decl);

        void finalize();

        const acontext_t &acontext() const { return cgctx.actx; }

        const mcontext_t &mcontext() const { return cgctx.mctx; }
        mcontext_t &mcontext() { return cgctx.mctx; }

    private:

        bool should_emit_function(clang::GlobalDecl decl);

        operation build_global_function_declaration(clang::GlobalDecl decl);

        operation build_global_definition(clang::GlobalDecl decl);
        operation build_global_function_definition(clang::GlobalDecl decl);
        operation build_global_var_definition(const clang::VarDecl *decl, bool tentative = false);

        hl::FuncOp build_function_body(hl::FuncOp fn, clang::GlobalDecl decl);

        hl::FuncOp emit_function_epilogue(hl::FuncOp fn, clang::GlobalDecl decl);

        void deal_with_missing_return(hl::FuncOp fn, const clang::FunctionDecl *decl);

        // Emit any needed decls for which code generation was deferred.
        void build_deferred();

        // Helper for `build_deferred` to apply actual codegen.
        operation build_global_decl(const clang::GlobalDecl &decl);
        // Emit code for a single global function or var decl.
        // Forward declarations are emitted lazily.
        operation build_global(clang::GlobalDecl decl);

        bool may_drop_function_return(clang::QualType rty) const;

        const std::vector< clang::GlobalDecl >& deferred_decls_to_emit() const;
        const std::map< mangled_name_ref, clang::GlobalDecl >& deferred_decls() const;

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

        // FIXME: should we use llvm::TrackingVH<mlir::Operation> here?
        using replacements_map = llvm::StringMap< mlir::Operation * >;
        replacements_map replacements;

        void add_replacement(string_ref name, mlir::Operation *Op);

        // Call replaceAllUsesWith on all pairs in replacements.
        void apply_replacements();

        inline auto lang() const { return acontext().getLangOpts(); }

        codegen_context &cgctx;
        cc::action_options &opts;
        const cc::vast_args &vargs;

        unsigned deferred_top_level_decls = 0;

        friend struct defer_handle_of_top_level_decl;
        llvm::SmallVector< clang::FunctionDecl *, 8 > deferred_inline_member_func_defs;

        meta_generator_ptr meta;
        default_codegen codegen;
    };

} // namespace vast::cg
