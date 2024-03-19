// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultDeclVisitor.hpp"

namespace vast::cg
{
    bool unsupported(const clang::FunctionDecl *decl) {
        if (decl->getAttr< clang::ConstructorAttr >()) {
            return true;
        }

        if (decl->getAttr< clang::DestructorAttr >()) {
            return true;
        }

        if (decl->isMultiVersion()) {
            return true;
        }

        if (llvm::dyn_cast< clang::CXXMethodDecl >(decl)) {
            return true; // Unsupported
        }

        return false;
    }

    operation default_decl_visitor::visit_prototype(const clang::FunctionDecl *decl) {
        if (unsupported(decl)) {
            return {};
        }

        return {};
    }

    operation default_decl_visitor::VisitFunctionDecl(const clang::FunctionDecl *decl) {
        if (unsupported(decl)) {
            return {};
        }

        return {};
    }

} // namespace vast::hl
