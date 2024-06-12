// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/Tower.hpp"

namespace vast::tw {
    auto default_loc_rewriter_t::insert(mlir::Operation *op) -> void {
        auto ctx = op->getContext();
        auto ol  = mlir::OpaqueLoc::get< mlir::Operation  *>(op, ctx);
        op->setLoc(mlir::FusedLoc::get({ op->getLoc() }, ol, ctx));
    }

    auto default_loc_rewriter_t::remove(mlir::Operation *op) -> void {
        auto fl = mlir::cast< mlir::FusedLoc >(op->getLoc());
        op->setLoc(fl.getLocations().front());
    }

    auto default_loc_rewriter_t::prev(mlir::Operation *op) -> mlir::Operation * {
        auto fl = mlir::cast< mlir::FusedLoc >(op->getLoc());
        auto ol = mlir::cast< mlir::OpaqueLoc >(fl.getMetadata());
        return mlir::OpaqueLoc::getUnderlyingLocation< mlir::Operation * >(ol);
    }

    loc_t location_maker::next(operation op) {
        auto mctx = op->getContext();
        auto id  = mlir::OpaqueLoc::get< operation >(op, mctx);
        return mlir::FusedLoc::get({ op->getLoc() }, id, mctx);
    }

    bool location_maker::location_query::are_tied(operation high, operation low) {
        return self(high) == prev(low);
    }
} // namespace vast::tw
