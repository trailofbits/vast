// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/LocationInfo.hpp"

namespace vast::tw {

    std::string location_info_t::fingerprint(const conversion_path_t &path) {
        std::string out;
        for (const auto &p : path) {
            out += p;
        }
        return out;
    }

    loc_t location_info_t::mk_unique_loc(const conversion_path_t &path, operation op) {
        auto mctx = op->getContext();
        auto raw_id =
            fingerprint(path) + std::to_string(reinterpret_cast< std::uintptr_t >(op));
        return mlir::FileLineColLoc::get(mctx, raw_id, 0, 0);
    }

    loc_t location_info_t::mk_linked_loc(loc_t self, loc_t prev) {
        auto mctx = self->getContext();
        return mlir::FusedLoc::get({ self, prev }, {}, mctx);
    }

    loc_t location_info_t::get_next(const conversion_path_t &path, operation op) {
        return mk_linked_loc(self(op), mk_unique_loc(path, op));
    }

    loc_t location_info_t::get_root(operation op) {
        return mk_linked_loc(op->getLoc(), mk_unique_loc({}, op));
    }

    bool location_info_t::are_tied(operation high, operation low) {
        return self(high) == prev(low);
    }

    void make_root(location_info_t &li, operation root) {
        auto set_loc = [&](operation op) { op->setLoc(li.get_root(op)); };
        root->walk(set_loc);
    }

    void transform_locations(location_info_t &li, const conversion_path_t &path, operation root) {
        auto set_loc = [&](operation op) { op->setLoc(li.get_next(path, op)); };
        root->walk(set_loc);
    }

} // namespace vast::tw
