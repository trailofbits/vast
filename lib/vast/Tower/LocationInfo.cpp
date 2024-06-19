// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/LocationInfo.hpp"

namespace vast::tw {
    loc_t location_info::get_next(operation high, operation low) {
        auto mctx = low->getContext();
        auto id  = mlir::OpaqueLoc::get< operation >(low, mctx);
        return mlir::FusedLoc::get({ self(high), id }, {}, mctx);
    }

    loc_t location_info::get_root(operation op) {
        auto mctx = op->getContext();
        auto id = mlir::OpaqueLoc::get< operation >(op, mctx);
        return mlir::FusedLoc::get({ op->getLoc(), id }, {}, mctx);
    }

    bool location_info::are_tied(operation high, operation low) {
        return self(high) == prev(low);
    }

    void make_root(location_info &li, operation root) {
        auto set_loc = [&](operation op) { op->setLoc(li.get_root(op)); };
        root->walk(set_loc);
    }

} // namespace vast::tw
