// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Generator.hpp"

namespace vast::cg {

    void vast_generator::anchor() {}

    void vast_generator::Initialize(acontext_t &actx) {
        this->acontext = &actx;
        this->mcontext = std::make_unique< mcontext_t >();

        codegen_driver_options options {
            .verbose_diagnostics = true,
            // forwarded options form clang codegen
            .coverage_mapping   = bool(cgo.CoverageMapping),
            .keep_static_consts = bool(cgo.KeepStaticConsts)
        };

        // TODO initialize dialects here
        this->codegen = std::make_unique< codegen_driver >(
            *acontext, *mcontext, options
        );
    }

    std::unique_ptr< mcontext_t > vast_generator::take_context() {
        return std::move(mcontext);
    }

    bool vast_generator::HandleTopLevelDecl(clang::DeclGroupRef decls) {
        if (diags.hasErrorOccurred())
            return true;

        return codegen->handle_top_level_decl(decls), true;
    }

    void vast_generator::HandleTranslationUnit(acontext_t &acontext) {
        codegen->handle_translation_unit(acontext);
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

    bool vast_generator::verify_module() const { return codegen->verify_module(); }

    owning_module_ref vast_generator::freeze() { return codegen->freeze(); }

} // namespace vast::cc
