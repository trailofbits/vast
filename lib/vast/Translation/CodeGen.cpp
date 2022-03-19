// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Translation/CodeGen.hpp"

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/Context.hpp"
#include "vast/Translation/HighLevelVisitor.hpp"

namespace vast::hl
{
    using builder_t = mlir::Builder;

    module_owning_ref high_level_codegen::emit_module(clang::Decl* decl) {

        builder_t bld(ctx);
        auto loc = bld.getUnknownLoc();
        module_owning_ref mod = module_t::create(loc);

        TranslationContext tctx(*ctx, decl->getASTContext(), mod);

        llvm::ScopedHashTableScope type_def_scope(tctx.type_defs);
        llvm::ScopedHashTableScope type_dec_scope(tctx.type_decls);
        llvm::ScopedHashTableScope enum_dec_scope(tctx.enum_decls);
        llvm::ScopedHashTableScope func_scope(tctx.functions);

        CodeGenVisitor visitor(tctx);
        visitor.Visit(decl);

        // TODO(Heno): verify module
        return mod;
    }

    bool high_level_codegen::emit_module(module_owning_ref &mod, clang::Decl* decl) {

        TranslationContext tctx(*ctx, decl->getASTContext(), mod);

        llvm::ScopedHashTableScope type_def_scope(tctx.type_defs);
        llvm::ScopedHashTableScope type_dec_scope(tctx.type_decls);
        llvm::ScopedHashTableScope enum_dec_scope(tctx.enum_decls);
        llvm::ScopedHashTableScope func_scope(tctx.functions);

        // TODO(akshayk): Need to populate scope variables with the
        //                 one available in the module

        CodeGenVisitor visitor(tctx);
        visitor.Visit(decl);

        // TODO: Verify module and return false if it fails
        return true;
    }


} // namespace vast::hl
