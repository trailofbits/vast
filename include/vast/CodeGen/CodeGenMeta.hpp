// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/AST/CXXInheritance.h>
#include <clang/AST/TypeLoc.h>
#include <clang/Basic/FileEntry.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Meta/MetaAttributes.hpp"

#include <concepts>

namespace vast::cg
{
    struct meta_generator {
        virtual ~meta_generator() = default;
        virtual loc_t location(const clang::Decl *) const = 0;
        virtual loc_t location(const clang::Stmt *) const = 0;
        virtual loc_t location(const clang::Expr *) const = 0;
    };

    struct default_meta_gen : meta_generator {
        default_meta_gen(acontext_t *actx, mcontext_t *mctx)
            : actx(actx), mctx(mctx)
        {}

        loc_t location(const clang::Decl *decl) const final {
            return location(decl->getLocation());
        }

        loc_t location(const clang::Stmt *stmt) const final {
            return location(stmt->getBeginLoc());
        }

        loc_t location(const clang::Expr *expr) const final {
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

    struct id_meta_gen : meta_generator {
        id_meta_gen(acontext_t *, mcontext_t *mctx)
            : mctx(mctx)
        {}

        loc_t location(const clang::Decl *decl) const final { return location_impl(decl); }
        loc_t location(const clang::Stmt *stmt) const final { return location_impl(stmt); }
        loc_t location(const clang::Expr *expr) const final { return location_impl(expr); }

      private:

        loc_t make_location(meta::IdentifierAttr id) const {
            auto dummy = mlir::UnknownLoc::get(mctx);
            return mlir::FusedLoc::get( { dummy }, id, mctx );
        }

        loc_t make_location(meta::identifier_t id) const {
            return make_location(meta::IdentifierAttr::get(mctx, id));
        }

        loc_t location_impl(auto token) const { return { make_location(counter++) }; }

        mutable meta::identifier_t counter = 0;

        mcontext_t *mctx;
    };

} // namespace vast::cg
