// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Location.h"
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenMetaGenerator.hpp"
#include "vast/CodeGen/Common.hpp"

namespace vast::cg {
    struct invalid_meta_gen final : meta_generator
    {
        invalid_meta_gen(mcontext_t *mctx) : mctx(mctx) {}

        loc_t location(const clang_decl *decl) const override {
            return mlir::UnknownLoc::get(mctx);
        }

        loc_t location(const clang_stmt *stmt) const override {
            return mlir::UnknownLoc::get(mctx);
        }

        loc_t location(const clang_expr *expr) const override {
            return mlir::UnknownLoc::get(mctx);
        }

      private:
        mcontext_t *mctx;
    };

} // namespace vast::cg
