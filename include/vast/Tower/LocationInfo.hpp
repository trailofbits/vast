// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Tower/Handle.hpp"

namespace vast::tw {

    // Is allowed to have state?
    struct location_info_t
    {
      private:
        // Encoded as `mlir::FusedLocation(original, mlir::OpaqueLocation(pointer_to_self))`
        using raw_loc_t = mlir::FusedLoc;

        static raw_loc_t raw_loc(operation op) {
            auto raw = mlir::dyn_cast< raw_loc_t >(op->getLoc());
            VAST_CHECK(raw, "{0} with loc: {1}", *op, op->getLoc());
            return raw;
        }

        template< std::size_t idx >
            requires (idx < 2)
        static loc_t get(raw_loc_t raw) {
            auto locs = raw.getLocations();
            VAST_ASSERT(locs.size() == 2);
            return locs[idx];
        }

        static auto parse(operation op) { return std::make_tuple(prev(op), self(op)); }

        // TODO: These are strictly not needed in this form, but help initial
        //       debugging a lot.
        std::string fingerprint(const conversion_path_t &);
        loc_t mk_loc(const conversion_path_t &, operation);

      public:
        // For the given operation return location to be used in this module.
        loc_t get_next(const conversion_path_t &, operation op);
        loc_t get_root(operation op);

        static loc_t self(raw_loc_t raw) { return get< 1 >(raw); }

        static loc_t prev(raw_loc_t raw) { return get< 0 >(raw); }

        static loc_t self(operation op) { return self(raw_loc(op)); }

        static loc_t prev(operation op) { return prev(raw_loc(op)); }

        static bool are_tied(operation high, operation low);
    };

    // Since we are going to tie together arbitrary modules, it makes sense to make them
    // have locations in the same shape - therefore root shouldn't be an excuse. It will
    // however require slightly different handling, so we are exposing a hook for that.
    void make_root(location_info_t &, operation);

    void transform_locations(location_info_t &, const conversion_path_t &, operation);

} // namespace vast::tw
