// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Generator.hpp"

namespace vast::cc {

    void vast_generator::anchor() {}

    void vast_generator::Initialize(AContext &/* acontext */) {
        throw compiler_error("Initialize not implemented");
    }

    bool vast_generator::HandleTopLevelDecl(clang::DeclGroupRef /* decl */) {
        throw compiler_error("HandleTopLevelDecl not implemented");
    }

    void vast_generator::HandleTranslationUnit(AContext &/* acontext */) {
        throw compiler_error("HandleTranslationUnit not implemented");
    }

    void vast_generator::HandleInlineFunctionDefinition(clang::FunctionDecl */* decl */) {
        throw compiler_error("HandleInlineFunctionDefinition not implemented");
    }

    void vast_generator::HandleTagDeclDefinition(clang::TagDecl */* decl */) {
        throw compiler_error("HandleTagDeclDefinition not implemented");
    }

    void vast_generator::HandleTagDeclRequiredDefinition(const clang::TagDecl */* decl */) {
        throw compiler_error("HandleTagDeclRequiredDefinition not implemented");
    }

} // namespace vast::cc
