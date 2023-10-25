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

#include "vast/CodeGen/CodeGenContext.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"

namespace vast::cc {

    using output_stream_ptr = std::unique_ptr< llvm::raw_pwrite_stream >;

    using clang_ast_consumer = clang::ASTConsumer;

    // TODO: Introduce helper wrapper on top of `vast_args`?
    enum class target_dialect { high_level, low_level, llvm };

    using backend = clang::BackendAction;

    struct vast_consumer : clang_ast_consumer
    {
        vast_consumer(
            output_type act, action_options opts,
            const vast_args &vargs, output_stream_ptr os
        )
            : action(act)
            , opts(std::move(opts))
            , vargs(vargs)
            , output_stream(std::move(os))
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

        void CompleteExternalDeclaration(clang::VarDecl * /* decl */) override;

        void AssignInheritanceModel(clang::CXXRecordDecl * /* decl */) override;

        void HandleVTable(clang::CXXRecordDecl * /* decl */) override;

      private:

        void emit_backend_output(
            backend backend_action, owning_module_ref mlir_module, mcontext_t *mctx
        );

        void emit_mlir_output(target_dialect target, owning_module_ref mod, mcontext_t *mctx);

        void compile_via_vast(vast_module mod, mcontext_t *mctx);

        virtual void anchor() {}

        output_type action;

        action_options opts;

        //
        // vast options
        //
        const vast_args &vargs;
        output_stream_ptr output_stream;

        //
        // contexts
        //
        std::unique_ptr< mcontext_t > mctx = nullptr;
        std::unique_ptr< cg::codegen_context > cgctx = nullptr;
        std::unique_ptr< cg::codegen_driver > codegen = nullptr;
    };
} // namespace vast::cc
