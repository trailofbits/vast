// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/LocationInfo.hpp"

namespace vast::tw {
    loc_t location_info::next(operation op) {
        auto mctx = op->getContext();
        auto id  = mlir::OpaqueLoc::get< operation >(op, mctx);
        return mlir::FusedLoc::get({ self(op), id }, {}, mctx);
    }

    bool location_info::are_tied(operation high, operation low) {
        return self(high) == prev(low);
    }

} // namespace vast::tw
