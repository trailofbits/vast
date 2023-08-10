// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Region.hpp"

namespace vast {

    Region* build_region(Builder &bld, State &st, BuilderCallback callback) {
        auto reg = st.addRegion();
        if (callback.has_value()) {
            bld.createBlock(reg);
            callback.value()(bld, st.location);
        }
        return reg;
    }

    Region* build_empty_region(Builder &bld, State &st) {
        auto reg = st.addRegion();
        reg->emplaceBlock();
        return reg;
    }

    hl::ValueYieldOp get_maybe_yield(Region &reg) {
        return mlir::dyn_cast< hl::ValueYieldOp >(reg.back().back());
    }

    hl::ValueYieldOp get_yield(Region &reg) {
        auto op = get_maybe_yield(reg);
        VAST_ASSERT(op && "getting yield from non-value region");
        return op;
    }

    mlir_value get_maybe_yielded_value(Region &reg) {
        if (auto yield = get_maybe_yield(reg))
            return yield.getResult();
        return {};
    }

    mlir_value get_yielded_value(Region &reg) {
        auto val = get_maybe_yielded_value(reg);
        VAST_ASSERT(val && "getting value from non-value region");
        return val;

    }

    mlir_type get_maybe_yielded_type(Region &reg) {
        if (auto val = get_maybe_yielded_value(reg))
            return val.getType();
        return {};
    }

    mlir_type get_yielded_type(Region &reg) {
        auto ty = get_maybe_yielded_type(reg);
        VAST_ASSERT(ty && "getting type from non-value region");
        return ty;
    }

    mlir::RegionSuccessor trivial_region_succ(Region *reg) {
        return { reg, reg->getArguments() };
    }

} // namespace vast
