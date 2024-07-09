// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/UnsupportedVisitor.hpp"

namespace vast::cg
{
    std::string decl_name(const clang_decl *decl) {
        std::stringstream ss;
        ss << decl->getDeclKindName();
        if (auto named = dyn_cast< clang::NamedDecl >(decl)) {
            ss << "::" << named->getNameAsString();
        }
        return ss.str();
    }

} // namespace vast::cg
