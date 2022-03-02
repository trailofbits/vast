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

    module_owning_ref HighLevelEmitter::emit_module(clang::Decl* decl) {
        builder_t bld(ctx);
        module_owning_ref mod = module_t::create(bld.getUnknownLoc());

        TranslationContext tctx(*ctx, decl->getASTContext(), mod);

        CodeGenVisitor visitor(tctx);
        visitor.Visit(decl);

        // TODO(Heno): verify module
        return mod;
    }

} // namespace vast::hl
