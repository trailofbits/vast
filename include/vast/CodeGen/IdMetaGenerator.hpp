// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/Common.hpp"
#include "vast/Dialect/Meta/MetaAttributes.hpp"
#include "vast/CodeGen/CodeGenMetaGenerator.hpp"

namespace vast::cg
{
    struct id_meta_gen final : meta_generator {
        id_meta_gen(acontext_t *, mcontext_t *mctx)
            : mctx(mctx)
        {}

        loc_t location(const clang_decl *decl) const override { return location_impl(decl); }
        loc_t location(const clang_stmt *stmt) const override { return location_impl(stmt); }
        loc_t location(const clang_expr *expr) const override { return location_impl(expr); }

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
