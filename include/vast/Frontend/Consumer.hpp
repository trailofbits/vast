// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTConsumer.h>
#include <clang/CodeGen/BackendUtil.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Diagnostics.hpp"
#include "vast/Frontend/FrontendAction.hpp"
#include "vast/Frontend/Options.hpp"
#include "vast/Frontend/Targets.hpp"

#include "vast/CodeGen/CodeGenDriver.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

namespace vast::cc {

    using output_stream_ptr = std::unique_ptr< llvm::raw_pwrite_stream >;

    using clang_ast_consumer = clang::ASTConsumer;

    using backend = clang::BackendAction;

    struct vast_consumer : clang_ast_consumer
    {
        vast_consumer(action_options opts, const vast_args &vargs, mcontext_t &mctx)
            : opts(std::move(opts)), vargs(vargs), mctx(mctx)
        {}

        void Initialize(acontext_t &ctx) override;

        bool HandleTopLevelDecl(clang::DeclGroupRef decls) override;

        void HandleCXXStaticMemberVarInstantiation(clang::VarDecl * /* decl */) override;

        void HandleInlineFunctionDefinition(clang::FunctionDecl * /* decl */) override;

        void HandleInterestingDecl(clang::DeclGroupRef /* decl */) override;

        void HandleTranslationUnit(acontext_t &acontext) override;

        void HandleTagDeclDefinition(clang::TagDecl *decl) override;

        // void HandleTagDeclRequiredDefinition(clang::TagDecl */* decl */) override {
        //     VAST_UNIMPLEMENTED;
        // }

        void CompleteTentativeDefinition(clang::VarDecl *decl) override;

        void AssignInheritanceModel(clang::CXXRecordDecl * /* decl */) override;

        void HandleVTable(clang::CXXRecordDecl * /* decl */) override;

        owning_mlir_module_ref result();

      protected:

        virtual void anchor() {}

        action_options opts;

        //
        // vast options
        //
        const vast_args &vargs;

        //
        // MLIR context.
        //
        mcontext_t &mctx;

        //
        // vast driver
        //
        std::unique_ptr< cg::driver > driver = nullptr;
    };

    struct vast_stream_consumer : vast_consumer {
        using base = vast_consumer;

        vast_stream_consumer(
            output_type act, action_options opts,
            const vast_args &vargs, mcontext_t &mctx,
            output_stream_ptr os
        )
            : base(std::move(opts), vargs, mctx), action(act), output_stream(std::move(os))
        {}

        void HandleTranslationUnit(acontext_t &acontext) override;

      private:
        void emit_backend_output(backend backend_action, owning_mlir_module_ref mod);

        void emit_mlir_output(target_dialect target, owning_mlir_module_ref mod);

        void process_mlir_module(target_dialect target, mlir_module mod);

        void print_mlir_bytecode(owning_mlir_module_ref mod);
        void print_mlir_string_format(owning_mlir_module_ref mod);

        output_type action;
        output_stream_ptr output_stream;
    };

} // namespace vast::cc
