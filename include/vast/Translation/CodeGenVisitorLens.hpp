// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/CodeGenVisitorBase.hpp"
#include "vast/Translation/Util.hpp"

namespace vast::hl {

    //
    // CodeGenVisitorLens
    //
    // Allows to access the visitor base from mixins.
    //
    template< typename Base, typename Derived >
    struct CodeGenVisitorLens
    {
        auto derived()       -> Derived      & { return static_cast<Derived&>(*this); }
        auto derived() const -> Derived const& { return static_cast<Derived const&>(*this); }

        auto context()       -> CodeGenContext      & { return derived().ctx; }
        auto context() const -> CodeGenContext const& { return derived().ctx; }

        auto mcontext()       -> MContext      & { return context().mctx; }
        auto mcontext() const -> MContext const& { return context().mctx; }

        auto acontext()       -> AContext      & { return context().actx; }
        auto acontext() const -> AContext const& { return context().actx; }

        auto &      meta_gen()       { return derived().meta; }
        const auto &meta_gen() const { return derived().meta; }

        template< typename Token >
        mlir::Location meta_location(Token token) const {
            return meta_gen().get(token).location();
        }

        template< typename Token >
        auto visit(Token token) { return derived().Visit(token); }

        template< typename Token >
        Type visit_as_lvalue_type(Token token) { return derived().VisitLValueType(token); }
    };

} // namespace vast::hl
