// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenMembers.hpp"

namespace vast::cg
{
    void members_generator::emit(const clang::RecordDecl *record) {
        for (auto *decl : record->decls()) {
            visitor.visit(decl);
        }
    }

} // namespace vast::cg
