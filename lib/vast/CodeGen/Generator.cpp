// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Generator.hpp"

namespace vast::cg {

    void vast_generator::anchor() {}

    void vast_generator::Initialize(acontext_t &acontext) {
        this->acontext = &acontext;

        mcontext = std::make_unique<mlir::MLIRContext>();
        // TODO initialize dialects here
        // CGM = std::make_unique<CIRGenModule>(*mlirCtx.get(), astCtx, codeGenOpts,
        //                                     Diags);
    }

    bool vast_generator::HandleTopLevelDecl(clang::DeclGroupRef /* decl */) {
        throw cc::compiler_error("HandleTopLevelDecl not implemented");
    }

    void vast_generator::HandleTranslationUnit(acontext_t &/* acontext */) {
        // Release the builder when there is no error.
        if (!diags.hasErrorOccurred() && cgm) {
            cgm->release();
        }

        // If there are errors before or when releasing the CGM, reset the module to
        // stop here before invoking the backend.
        if (diags.hasErrorOccurred()) {
            if (cgm) {
                // TODO: CGM->clear();
                // TODO: M.reset();
                return;
            }
        }
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
