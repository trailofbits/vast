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
#include "vast/Translation/CodeGenDriver.hpp"
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

        owning_module_ref freeze();
        std::unique_ptr< mcontext_t > take_context();

        bool verify_module() const;

        const target_info_t &get_target_info();

        void dump_module() { codegen->dump_module(); }

    protected:
        std::unique_ptr< codegen_driver > codegen = nullptr;
        std::unique_ptr< mcontext_t > mcontext = nullptr;

    private:
        virtual void anchor();

        cc::diagnostics_engine &diags;
        acontext_t *acontext;

        mutable std::unique_ptr< target_info_t > target_info;

        const cc::codegen_options cgo; // intentionally copied
    };

} // namespace vast::cg
