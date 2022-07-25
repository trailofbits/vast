// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Translation/CodeGen.hpp"

namespace vast::hl
{
    OwningModuleRef CodeGen::emit_module(
        clang::ASTUnit *unit, const TranslationConfig &cfg
    ) {
        return process_ast(unit, cfg);
    }

    static bool top_level_decl_process(void * context, const clang::Decl *decl) {
        CodeGenVisitor &visitor = *static_cast<CodeGenVisitor*>(context);
        return visitor.Visit(const_cast< clang::Decl * >(decl)), true;
    }

    void CodeGen::process(clang::ASTUnit *unit, CodeGenVisitor &visitor) {
        unit->visitLocalTopLevelDecls(&visitor, top_level_decl_process);
    }

    OwningModuleRef CodeGen::emit_module(
        clang::Decl* decl, const TranslationConfig &cfg
    ) {
        return process_ast(decl, cfg);
    }

    void CodeGen::process(clang::Decl *decl, CodeGenVisitor &visitor) {
        visitor.Visit(decl);
    }


} // namespace vast::hl
