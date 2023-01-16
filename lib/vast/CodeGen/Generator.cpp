// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Generator.hpp"

namespace vast::cg {

    void vast_generator::anchor() {}

    void vast_generator::Initialize(AContext &acontext) {
        this->acontext = &acontext;

        mcontext = std::make_unique<mlir::MLIRContext>();
        // TODO initialize dialects here
        // CGM = std::make_unique<CIRGenModule>(*mlirCtx.get(), astCtx, codeGenOpts,
        //                                     Diags);
    }

    bool vast_generator::HandleTopLevelDecl(clang::DeclGroupRef /* decl */) {
        throw cc::compiler_error("HandleTopLevelDecl not implemented");
    }

    void vast_generator::HandleTranslationUnit(AContext &/* acontext */) {
        throw cc::compiler_error("HandleTranslationUnit not implemented");
    }

    void vast_generator::HandleInlineFunctionDefinition(clang::FunctionDecl */* decl */) {
        throw cc::compiler_error("HandleInlineFunctionDefinition not implemented");
    }

    void vast_generator::HandleTagDeclDefinition(clang::TagDecl */* decl */) {
        throw cc::compiler_error("HandleTagDeclDefinition not implemented");
    }

    void vast_generator::HandleTagDeclRequiredDefinition(const clang::TagDecl */* decl */) {
        throw cc::compiler_error("HandleTagDeclRequiredDefinition not implemented");
    }

} // namespace vast::cc
