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
        visitor.visit(decl);
    }

    void module_generator::finalize() { scope().finalize(); }

} // namespace vast::cg
