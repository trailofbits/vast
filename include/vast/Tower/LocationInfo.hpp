// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

namespace vast::tw {

    // Is allowed to have state?
    struct location_info {
      private:
        // Encoded as `mlir::FusedLocation(original, mlir::OpaqueLocation(pointer_to_self))`
        using raw_loc_t = mlir::FusedLoc;

        static raw_loc_t raw_loc(operation op) {
            auto raw = mlir::dyn_cast< raw_loc_t >(op->getLoc());
            VAST_ASSERT(raw);
            return raw;
        }

        template< std::size_t idx > requires (idx < 2)
        static loc_t get(raw_loc_t raw) {
            auto locs = raw.getLocations();
            VAST_ASSERT(locs.size() == 2);
            return locs[idx];
        }

        static loc_t prev(raw_loc_t raw) { return get< 0 >(raw); }
        static loc_t self(raw_loc_t raw) { return get< 1 >(raw); }

        static loc_t prev(operation op) { return prev(raw_loc(op)); }
        static loc_t self(operation op) { return self(raw_loc(op)); }

        static auto parse(operation op) {
            return std::make_tuple(prev(op), self(op));
        }

      public:
        // For the given operation return location to be used in this module.
        loc_t next(operation low_op);
        static bool are_tied(operation high, operation low);
    };

} // namespace vast::tw
