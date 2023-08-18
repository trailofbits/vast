// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg {

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

        //
        // Contexts
        //
        auto context()       -> auto      & { return derived().ctx; }
        auto context() const -> auto const& { return derived().ctx; }

        auto mcontext()       -> mcontext_t      & { return context().mctx; }
        auto mcontext() const -> mcontext_t const& { return context().mctx; }

        auto acontext()       -> acontext_t      & { return context().actx; }
        auto acontext() const -> acontext_t const& { return context().actx; }

        auto name_mangler()       -> CodeGenMangler      & { return context().mangler; }
        auto name_mangler() const -> CodeGenMangler const& { return context().mangler; }

        //
        // meta
        //
        auto &      meta_gen()       { return derived().meta; }
        const auto &meta_gen() const { return derived().meta; }

        template< typename Token >
        mlir::Location meta_location(Token token) const {
            return meta_gen().get(token).location();
        }

        template< typename Token >
        auto visit(Token token) { return derived().Visit(token); }

        template< typename Token >
        mlir_type visit_as_lvalue_type(Token token) { return derived().VisitLValueType(token); }
    };

} // namespace vast::cg
