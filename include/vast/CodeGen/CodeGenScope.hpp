// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Value.h>
#include <mlir/IR/Builders.h>
#include <llvm/ADT/SmallPtrSet.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast::cg
{
    struct CodeGenContext;

    // Represents a vast.scope, vast.if, and then/else regions. i.e. lexical
    // scopes that require cleanups.
    struct lexical_scope_context {
      private:
        // Block containing cleanup code for things initialized in this
        // lexical context (scope).
        mlir::Block *cleanup_block = nullptr;

        // Points to scope entry block. This is useful, for instance, for
        // helping to insert allocas before finalizing any recursive codegen
        // from switches.
        mlir::Block *entry_block;

        // On a coroutine body, the OnFallthrough sub stmt holds the handler
        // (CoreturnStmt) for control flow falling off the body. Keep track
        // of emitted co_return in this scope and allow OnFallthrough to be
        // skipeed.
        bool _has_coreturn = false;

        // FIXME: perhaps we can use some info encoded in operations.
        enum class kind
        {
            regular_scope, // vast.if, vast.scope, if_regions
            switch_scope   // vast.switch
        } scope_kind = kind::regular_scope;


      public:
        unsigned depth = 0;
        bool has_return = false;

        lexical_scope_context(mlir::Location b, mlir::Location e, mlir::Block *eb)
            : entry_block(eb)
            , begin_loc(b)
            , end_loc(e)
        {}

        ~lexical_scope_context() = default;

        // ---
        // Coroutine tracking
        // ---
        bool has_coreturn() const { return _has_coreturn; }
        void set_coreturn() { _has_coreturn = true; }

        // ---
        // Kind
        // ---
        bool is_regular() { return scope_kind == kind::regular_scope; }
        bool is_switch() { return scope_kind == kind::switch_scope; }
        void set_as_switch() { scope_kind = kind::switch_scope; }

        // ---
        // Goto handling
        // ---

        // Lazy create cleanup block or return what's available.
        mlir::Block *get_or_create_cleanup_block(mlir::OpBuilder &builder) {
            if (cleanup_block)
                return get_cleanup_block(builder);
            return create_cleanup_block(builder);
        }

        mlir::Block *get_cleanup_block(mlir::OpBuilder &/* builder */) { return cleanup_block; }
        mlir::Block *create_cleanup_block(mlir::OpBuilder &builder) {
            // Create the cleanup block but dont hook it up around just yet.
            InsertionGuard guard(builder);
            cleanup_block = builder.createBlock(builder.getBlock()->getParent());
            VAST_ASSERT(builder.getInsertionBlock());
            return cleanup_block;
        }

        // Goto's introduced in this scope but didn't get fixed.
        llvm::SmallVector< std::pair< operation, const clang::LabelDecl * >, 4 > pending_gotos;

        // Labels solved inside this scope.
        llvm::SmallPtrSet< const clang::LabelDecl *, 4 > solved_labels;

        // ---
        // Return handling
        // ---

      private:
        // On switches we need one return block per region, since cases don't
        // have their own scopes but are distinct regions nonetheless.
        llvm::SmallVector< mlir::Block * > ret_blocks;
        llvm::SmallVector< llvm::Optional< mlir::Location > > ret_locs;

        // There's usually only one ret block per scope, but this needs to be
        // get or create because of potential unreachable return statements, note
        // that for those, all source location maps to the first one found.
        mlir::Block *create_ret_block(mlir::Location /* loc */) {
            VAST_CHECK((is_switch() || ret_blocks.size() == 0), "only switches can hold more than one ret block");

            // Create the cleanup block but dont hook it up around just yet.
            // InsertionGuard guard(CGF.builder);
            // auto *b = CGF.builder.createBlock(CGF.builder.getBlock()->getParent());
            // ret_blocks.push_back(b);
            // ret_locs.push_back(loc);
            // return b;
            VAST_UNIMPLEMENTED;
        }

      public:
        llvm::ArrayRef< mlir::Block * > get_ret_blocks() { return ret_blocks; }
        llvm::ArrayRef< llvm::Optional< mlir::Location > > get_ret_locs() { return ret_locs; }

        mlir::Block *get_or_create_ret_block(mlir::Location loc) {
            unsigned int region_idx = 0;
            if (is_switch()) {
                VAST_UNIMPLEMENTED_MSG("switch region block");
            }
            if (region_idx >= ret_blocks.size()) {
                return create_ret_block(loc);
            }
            return &*ret_blocks.back();
        }

        // Scope entry block tracking
        mlir::Block *get_entry_block() { return entry_block; }

        mlir::Location begin_loc, end_loc;
    };

    template< typename codegen_t >
    class lexical_scope_guard {
        codegen_t &codegen;
        lexical_scope_context *old_val = nullptr;

    public:
        lexical_scope_guard(codegen_t &codegen, lexical_scope_context *ctx)
            : codegen(codegen)
        {
            if (codegen.current_lexical_scope()) {
                old_val = codegen.current_lexical_scope();
                ctx->depth++;
            }

            codegen.set_current_lexical_scope(ctx);
        }


        lexical_scope_guard(const lexical_scope_guard &) = delete;
        lexical_scope_guard &operator=(const lexical_scope_guard &) = delete;
        lexical_scope_guard &operator=(lexical_scope_guard &&other) = delete;

        ~lexical_scope_guard() { cleanup(); restore(); }

        void restore() { codegen.set_current_lexical_scope(old_val); }

        // All scope related cleanup needed:
        // - Patching up unsolved goto's.
        // - Build all cleanup code and insert yield/returns.
        void cleanup() {
            auto *local_scope = codegen.current_lexical_scope();

            // Handle pending gotos and the solved labels in this scope.
            if (!local_scope->pending_gotos.empty()) {
                VAST_UNIMPLEMENTED_MSG( "scope cleanup of unresolved gotos" );
            }
            local_scope->solved_labels.clear();
            // TODO clean up blocks
        }

  };

} // namespace vast::cg
