// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/AST/TypeLoc.h>
#include <clang/Basic/FileEntry.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Meta/MetaAttributes.hpp"

#include <concepts>

namespace vast::hl
{
    template< typename T >
    concept MetaLike = requires(T meta) {
        { meta.location() } -> std::convertible_to< mlir::Location >;
    };

    template< typename T >
    concept MetaGeneratorLike = requires(T gen) {
        { gen.get( std::declval< const clang::Decl * >() ) } -> MetaLike;
        { gen.get( std::declval< const clang::Stmt * >() ) } -> MetaLike;
        { gen.get( std::declval< const clang::Expr * >() ) } -> MetaLike;
        { gen.get( std::declval< const clang::Type * >() ) } -> MetaLike;
        { gen.get( std::declval< clang::QualType >() ) }     -> MetaLike;
    };

    struct DefaultMeta {
        mlir::Location location() const { return _location; }
        mlir::Location _location;
    };

    struct DefaultMetaGenerator {
        DefaultMetaGenerator(AContext *actx, MContext *mctx)
            : actx(actx), mctx(mctx)
        {}

        DefaultMeta get(const clang::FullSourceLoc &loc) const {
            auto file = loc.getFileEntry()->getName();
            auto line = loc.getLineNumber();
            auto col  = loc.getColumnNumber();
            return { mlir::FileLineColLoc::get(mctx, file, line, col) };
        }

        DefaultMeta get(const clang::SourceLocation &loc) const {
            return get(clang::FullSourceLoc(loc, actx->getSourceManager()));
        }

        DefaultMeta get(const clang::Decl *decl) const {
            return get(decl->getLocation());
        }

        DefaultMeta get(const clang::Stmt *stmt) const {
            // TODO: use SoureceRange
            return get(stmt->getBeginLoc());
        }

        DefaultMeta get(const clang::Expr *expr) const {
            // TODO: use SoureceRange
            return get(expr->getExprLoc());
        }

        DefaultMeta get(const clang::TypeLoc &loc) const {
            // TODO: use SoureceRange
            return get(loc.getBeginLoc());
        }

        DefaultMeta get(const clang::Type *type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        DefaultMeta get(clang::QualType type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        AContext *actx;
        MContext *mctx;
    };

} // namespace vast::hl
