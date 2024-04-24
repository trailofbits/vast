// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenModule.hpp"

#include "vast/CodeGen/DataLayout.hpp"
#include "vast/CodeGen/CodeGenFunction.hpp"

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/DefaultTypeVisitor.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

#include "vast/Util/Common.hpp"

namespace vast::cg
{
    //
    // Module Generator
    //
    void module_generator::emit(clang::DeclGroupRef decls) {
        for (auto &decl : decls) { emit(decl); }
    }

    void module_generator::emit(clang::Decl *decl) {
        switch (decl->getKind()) {
            case clang::Decl::Kind::Typedef:
                return emit(cast<clang::TypedefDecl>(decl));
            case clang::Decl::Kind::Enum:
                return emit(cast<clang::EnumDecl>(decl));
            case clang::Decl::Kind::Record:
                return emit(cast<clang::RecordDecl>(decl));
            case clang::Decl::Kind::Function:
                return emit(cast<clang::FunctionDecl>(decl));
            case clang::Decl::Kind::Var:
                return emit(cast<clang::VarDecl>(decl));
            default:
                VAST_FATAL("unhandled decl kind: {}", decl->getDeclKindName());
        }
    }

    void module_generator::emit(clang_global */* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::TypedefDecl *decl) {
        visitor.visit(decl);
    }

    void module_generator::emit(clang::EnumDecl *decl) {
        visitor.visit(decl);
    }

    void module_generator::emit(clang::RecordDecl *decl) {
        visitor.visit(decl);
    }

    void module_generator::emit(clang::FunctionDecl *decl) {
        auto gen = mk_scoped_generator< function_generator >(*this, opts);
        gen.emit(decl);
    }

    void module_generator::emit(clang::VarDecl *decl) {
        visitor.visit(decl);
    }


    void module_generator::finalize() { scope().finalize(); }

} // namespace vast::cg
