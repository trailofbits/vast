// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/AST/TypeLoc.h>
#include <clang/Basic/FileEntry.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/CodeGenMetaGenerator.hpp"

namespace vast::cg
{
    struct default_meta_gen final : meta_generator {
        default_meta_gen(acontext_t *actx, mcontext_t *mctx)
            : actx(actx), mctx(mctx)
        {}

        loc_t location(const clang_decl *decl) const override {
            return location(decl->getLocation());
        }

        loc_t location(const clang_stmt *stmt) const override {
            return location(stmt->getBeginLoc());
        }

        loc_t location(const clang_expr *expr) const override {
            return location(expr->getExprLoc());
        }

      private:

        loc_t location(const clang::FullSourceLoc &loc) const {
            auto file = loc.getFileEntryRef() ? loc.getFileEntryRef()->getName() : "unknown";
            auto line = loc.getLineNumber();
            auto col  = loc.getColumnNumber();
            return { mlir::FileLineColLoc::get(mctx, file, line, col) };
        }

        loc_t location(const clang::SourceLocation &loc) const {
            if (loc.isValid())
                return location(clang::FullSourceLoc(loc, actx->getSourceManager()));
            return mlir::UnknownLoc::get(mctx);
        }

        acontext_t *actx;
        mcontext_t *mctx;
    };

} // namespace vast::cg
