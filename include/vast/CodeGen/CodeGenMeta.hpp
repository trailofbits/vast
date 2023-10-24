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
    template< typename T >
    concept meta_like = requires(T meta) {
        { meta.location() } -> std::convertible_to< loc_t >;
    };

    template< typename T >
    concept meta_generator_like = requires(T gen) {
        { gen.get( std::declval< const clang::Decl * >() ) } -> meta_like;
        { gen.get( std::declval< const clang::Stmt * >() ) } -> meta_like;
        { gen.get( std::declval< const clang::Expr * >() ) } -> meta_like;
        { gen.get( std::declval< const clang::Type * >() ) } -> meta_like;
        { gen.get( std::declval< clang::QualType >() ) }     -> meta_like;
    };

    struct default_meta {
        loc_t location() const { return _location; }
        loc_t _location;
    };

    struct default_meta_gen {
        default_meta_gen(acontext_t *actx, mcontext_t *mctx)
            : actx(actx), mctx(mctx)
        {}

        default_meta get(const clang::FullSourceLoc &loc) const {
            auto file = loc.getFileEntry() ? loc.getFileEntry()->getName() : "unknown";
            auto line = loc.getLineNumber();
            auto col  = loc.getColumnNumber();
            return { mlir::FileLineColLoc::get(mctx, file, line, col) };
        }

        default_meta get(const clang::SourceLocation &loc) const {
            return get(clang::FullSourceLoc(loc, actx->getSourceManager()));
        }

        default_meta get(const clang::Decl *decl) const {
            return get(decl->getLocation());
        }

        default_meta get(const clang::Stmt *stmt) const {
            // TODO: use SoureceRange
            return get(stmt->getBeginLoc());
        }

        default_meta get(const clang::Expr *expr) const {
            // TODO: use SoureceRange
            return get(expr->getExprLoc());
        }

        default_meta get(const clang::TypeLoc &loc) const {
            // TODO: use SoureceRange
            return get(loc.getBeginLoc());
        }

        default_meta get(const clang::Type *type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        default_meta get(clang::QualType type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        default_meta get(const clang::CXXBaseSpecifier &spec) const {
            return get(spec.getBeginLoc());
        }

        acontext_t *actx;
        mcontext_t *mctx;
    };

    struct id_meta_gen {
        id_meta_gen(acontext_t *actx, mcontext_t *mctx)
            : actx(actx), mctx(mctx)
        {}

        loc_t make_location(meta::IdentifierAttr id) const {
            auto dummy = mlir::UnknownLoc::get(mctx);
            return mlir::FusedLoc::get( { dummy }, id, mctx );
        }

        loc_t make_location(meta::identifier_t id) const {
            return make_location(meta::IdentifierAttr::get(mctx, id));
        }

        default_meta get_impl(auto token) const { return { make_location(counter++) }; }

        default_meta get(const clang::Decl *decl) const { return get_impl(decl); }
        default_meta get(const clang::Stmt *stmt) const { return get_impl(stmt); }
        default_meta get(const clang::Expr *expr) const { return get_impl(expr); }
        default_meta get(const clang::Type *type) const { return get_impl(type); }
        default_meta get(clang::QualType type) const { return get_impl(type); }
        default_meta get(const clang::CXXBaseSpecifier &spec) const { return get_impl(spec); }

        mutable meta::identifier_t counter = 0;

        acontext_t *actx;
        mcontext_t *mctx;
    };

} // namespace vast::cg
