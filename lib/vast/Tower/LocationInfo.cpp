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

    loc_t location_info_t::mk_loc(const conversion_path_t &path, operation op) {
        auto mctx = op->getContext();
        auto raw_id =
            fingerprint(path) + std::to_string(reinterpret_cast< std::uintptr_t >(op));
        return mlir::FileLineColLoc::get(mctx, raw_id, 0, 0);
    }

    loc_t location_info_t::get_next(const conversion_path_t &path, operation op) {
        auto mctx = op->getContext();
        auto next = mlir::FusedLoc::get({ self(op), mk_loc(path, op) }, {}, mctx);

        return next;
    }

    loc_t location_info_t::get_root(operation op) {
        auto mctx = op->getContext();
        auto next = mlir::FusedLoc::get({ op->getLoc(), mk_loc({}, op) }, {}, mctx);
        return next;
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
