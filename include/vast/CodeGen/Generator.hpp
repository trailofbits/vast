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
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/Util/Common.hpp"

#include <memory>

namespace vast::cg {

    using clang_ast_consumer = clang::ASTConsumer;

    struct vast_generator : clang_ast_consumer {

        vast_generator(cc::diagnostics_engine &diags, const cc::codegen_options &cgo)
            : diags(diags), cgo(cgo)
        {}

        void Initialize(acontext_t &) override;

        bool HandleTopLevelDecl(clang::DeclGroupRef) override;
        void HandleTranslationUnit(acontext_t &) override;
        void HandleInlineFunctionDefinition(clang::FunctionDecl *) override;
        void HandleTagDeclDefinition(clang::TagDecl *) override;
        void HandleTagDeclRequiredDefinition(const clang::TagDecl *) override;
        void CompleteTentativeDefinition(clang::VarDecl *decl) override;

        owning_module_ref freeze();
        std::unique_ptr< mcontext_t > take_context();

        bool verify_module() const;

        target_info_t &get_target_info();
        type_info_t &get_type_info();

        void dump_module() { codegen->dump_module(); }

    protected:
        std::unique_ptr< mcontext_t > mcontext = nullptr;
        std::unique_ptr< CodeGenContext > cgcontext = nullptr;
        std::unique_ptr< codegen_driver > codegen = nullptr;

    private:
        virtual void anchor();

        cc::diagnostics_engine &diags;
        acontext_t *acontext;

        const cc::codegen_options cgo; // intentionally copied
    };

} // namespace vast::cg
