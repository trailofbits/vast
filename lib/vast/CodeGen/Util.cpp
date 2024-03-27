// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    std::string get_decl_name(const clang::NamedDecl *decl) {
        if (decl->getIdentifier())
            return decl->getName().str();
        return "anonymous[" + std::to_string(decl->getID()) + "]";
    }

    std::string get_namespaced_decl_name(const clang::NamedDecl *decl) {
        // gather contexts
        std::vector< const clang::DeclContext * > dctxs;
        for (const auto *dctx = decl->getDeclContext(); dctx; dctx = dctx->getParent()) {
            dctxs.push_back(dctx);
        }

        std::string name;
        for (const auto *dctx : llvm::reverse(dctxs)) {
            if (llvm::isa< clang::TranslationUnitDecl >(dctx))
                continue;

            if (llvm::isa< clang::FunctionDecl >(dctx))
                continue;

            if (llvm::isa< clang::LinkageSpecDecl >(dctx))
                continue;

            if (const auto *d = llvm::dyn_cast< clang::NamedDecl >(dctx)) {
                name += get_decl_name(d);
            } else {
                VAST_FATAL("unknown decl context: {0}", dctx->getDeclKindName());
            }

            name += "::";
        }

        return name;
    }

} // namespace vast::cg
