// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

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
        explicit DefaultMetaGenerator(MContext *ctx) : ctx(ctx) {}

        DefaultMeta get(const clang::Decl * /* decl */) const {
            return { mlir::UnknownLoc::get(ctx) };
        }

        DefaultMeta get(const clang::Stmt * /* stmt */) const {
            return { mlir::UnknownLoc::get(ctx) };
        }

        DefaultMeta get(const clang::Expr * /* expr */) const {
            return { mlir::UnknownLoc::get(ctx) };
        }

        DefaultMeta get(const clang::Type * /* type */) const {
            return { mlir::UnknownLoc::get(ctx) };
        }

        DefaultMeta get(clang::QualType /* type */) const {
            return { mlir::UnknownLoc::get(ctx) };
        }

        MContext *ctx;
    };

} // namespace vast::hl
