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
#include "vast/CodeGen/CodeGenTypeDriver.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

// FIXME: bringing dependency from upper layer
#include "vast/CodeGen/CXXABI.hpp"
#include "vast/CodeGen/TypeInfo.hpp"
#include "vast/CodeGen/TargetInfo.hpp"
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

    using target_info_ptr = std::unique_ptr< target_info_t >;

    x86_avx_abi_level avx_level(const clang::TargetInfo &target);

    namespace detail {
        target_info_ptr initialize_target_info(
            const clang::TargetInfo &target, const type_info_t &type_info
        );
    } // namespace detail

    // This is a layer that provides interface between
    // clang codegen and vast codegen

    struct codegen_driver {

        explicit codegen_driver(
            CodeGenContext &cgctx
            , codegen_options opts
        )
            : actx(cgctx.actx)
            , mctx(cgctx.mctx)
            , options(opts)
            , cxx_abi(create_cxx_abi(actx))
            , codegen(cgctx)
            , type_conv(*this)
        {
            type_info = std::make_unique< type_info_t >(*this);

            const auto &target = actx.getTargetInfo();
            target_info = detail::initialize_target_info(target, get_type_info());
        }

        ~codegen_driver() {
            VAST_ASSERT(deferred_inline_member_func_defs.empty());
        }

        codegen_driver(const codegen_driver &) = delete;
        codegen_driver(codegen_driver &&) = delete;

        codegen_driver& operator=(const codegen_driver &) = delete;
        codegen_driver& operator=(codegen_driver &&) = delete;

        bool verify_module() const;

        void handle_translation_unit(acontext_t &acontext);
        void handle_top_level_decl(clang::DeclGroupRef decls);
        void handle_top_level_decl(clang::Decl *decl);

        void finalize();

        const target_info_t &get_target_info() const { return *target_info; }
        target_info_t &get_target_info() { return *target_info; }

        const type_info_t &get_type_info() const { return *type_info; }
        type_info_t &get_type_info() { return *type_info; }

        const acontext_t &acontext() const { return actx; }

        const mcontext_t &mcontext() const { return mctx; }
        mcontext_t &mcontext() { return mctx; }

        function_processing_lock make_lock(const function_info_t *fninfo);

        void update_completed_type(const clang::TagDecl *tag);

        friend struct type_conversion_driver;

        mangled_name_ref get_mangled_name(clang::GlobalDecl decl);

        void dump_module() { codegen.dump_module(); }

    private:

        bool should_emit_function(clang::GlobalDecl decl);

        vast_cxx_abi get_cxx_abi() const;

        static vast_cxx_abi *create_cxx_abi(const acontext_t &actx) {
            switch (actx.getCXXABIKind()) {
                case clang::TargetCXXABI::GenericItanium:
                case clang::TargetCXXABI::GenericAArch64:
                case clang::TargetCXXABI::AppleARM64:
                    return create_vast_itanium_cxx_abi(actx);
                default:
                    VAST_UNREACHABLE("invalid C++ ABI kind");
            }
        }

        CodeGenContext::VarTable &variables_symbol_table();

        bool has_this_return(clang::GlobalDecl decl) const;
        bool has_most_derived_return(clang::GlobalDecl decl) const;

        operation build_global_definition(clang::GlobalDecl decl);
        operation build_global_function_definition(clang::GlobalDecl decl);
        operation build_global_var_definition(const clang::VarDecl *decl, bool tentative = false);

        function_arg_list build_function_arg_list(clang::GlobalDecl decl);
        hl::FuncOp build_function_body(hl::FuncOp fn, clang::GlobalDecl decl, const function_info_t &fty_info);

        // Emit any needed decls for which code generation was deferred.
        void build_deferred();

        // Helper for `build_deferred` to apply actual codegen.
        operation build_global_decl(const clang::GlobalDecl &decl);
        // Emit code for a single global function or var decl.
        // Forward declarations are emitted lazily.
        operation build_global(clang::GlobalDecl decl);

        operation get_global_value(mangled_name_ref name);
        mlir_value get_global_value(const clang::Decl *decl);


        bool may_drop_function_return(qual_type rty) const;

        const std::vector< clang::GlobalDecl >& default_methods_to_emit() const;
        const std::vector< clang::GlobalDecl >& deferred_decls_to_emit() const;
        const std::vector< const clang::CXXRecordDecl * >& deferred_vtables() const;
        const std::map< mangled_name_ref, clang::GlobalDecl >& deferred_decls() const;

        std::vector< clang::GlobalDecl > receive_deferred_decls_to_emit();

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

        inline auto lang() const { return actx.getLangOpts(); }

        acontext_t &actx;
        mcontext_t &mctx;

        codegen_options options;

        unsigned deferred_top_level_decls = 0;

        friend struct defer_handle_of_top_level_decl;
        llvm::SmallVector< clang::FunctionDecl *, 8 > deferred_inline_member_func_defs;

        std::unique_ptr< vast_cxx_abi > cxx_abi;

        // FIXME: make configurable
        CodeGenWithMetaIDs codegen;

        mutable std::unique_ptr< target_info_t > target_info;
        mutable std::unique_ptr< type_info_t > type_info;

        type_conversion_driver type_conv;

    };

} // namespace vast::cg
