// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::cg
{
    struct codegen_driver_options {
        bool verbose_diagnostics = true;
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

    // This is a layer that provides interface between
    // clang codegen and vast codegen
    struct codegen_driver {

        explicit codegen_driver(acontext_t &/* actx */, mcontext_t &mctx)
            : mctx(mctx)
            , options{
                .verbose_diagnostics = true
            }
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
        bool handle_top_level_decl(clang::DeclGroupRef decls);
        bool handle_top_level_decl(clang::Decl *decl);

        vast_module take_module() { return std::move(mod); }

    private:

        void build_deferred_decls();
        void build_default_methods();

        mcontext_t &mctx;
        vast_module mod;

        codegen_driver_options options;

        unsigned deffered_top_level_decls = 0;

        friend struct defer_handle_of_top_level_decl;

        llvm::SmallVector< clang::FunctionDecl *, 8 > deferred_inline_member_func_defs;
    };

} // namespace vast::cg
