// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/Decl.h>
#include <clang/Basic/CodeGenOptions.h>
#include <llvm/Support/ToolOutputFile.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Diagnostics.hpp"
#include "vast/Util/Common.hpp"

#include <memory>

namespace vast::cg {

    using clang_ast_consumer    = clang::ASTConsumer;

    struct vast_generator : clang_ast_consumer {

        vast_generator(cc::diagnostics_engine &diags, const cc::codegen_options &cgo)
            : diags(diags), cgo(cgo)
        {}

        ~vast_generator() {
            // There should normally not be any leftover inline method definitions.
            assert(deferred_inline_member_func_defs.empty() || diags.hasErrorOccurred());
        }

        void Initialize(AContext &) override;

        bool HandleTopLevelDecl(clang::DeclGroupRef) override;
        void HandleTranslationUnit(AContext &) override;
        void HandleInlineFunctionDefinition(clang::FunctionDecl *) override;
        void HandleTagDeclDefinition(clang::TagDecl *) override;
        void HandleTagDeclRequiredDefinition(const clang::TagDecl *) override;

        // bool EmitFunction(const clang::FunctionDecl *FD);

        // mlir::ModuleOp getModule();
        // std::unique_ptr<mlir::MLIRContext> takeContext() {
        //     return std::move(mlirCtx);
        // };

        // bool verifyModule();

        // void buildDeferredDecls();
        // void buildDefaultMethods();

    protected:

        std::unique_ptr< MContext > mcontext;
        // std::unique_ptr< vast_module > mod;

    private:
        virtual void anchor();

        cc::diagnostics_engine &diags;
        AContext *acontext;

        const cc::codegen_options cgo; // intentionally copied
        // unsigned handling_pop_level_decls;

        llvm::SmallVector< clang::FunctionDecl *, 8>  deferred_inline_member_func_defs;
    };

} // namespace vast::cg
