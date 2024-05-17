// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Region.hpp"

namespace vast {

    region_t* build_region(mlir_builder &bld, op_state &st, maybe_builder_callback_ref callback) {
        return build_region(st.addRegion(), bld, st, callback);
    }

    region_t* build_region(mlir_builder &bld, op_state &st, builder_callback_ref callback) {
        return build_region(st.addRegion(), bld, st, callback);
    }

    region_t* build_region(region_t *reg, mlir_builder &bld, op_state &st, maybe_builder_callback_ref callback) {
        if (callback.has_value()) {
            build_region(reg, bld, st, callback.value());
        }
        return reg;
    }

    region_t* build_region(region_t *reg, mlir_builder &bld, op_state &st, builder_callback_ref callback) {
        bld.createBlock(reg);
        callback(bld, st.location);
        return reg;
    }

    region_t* build_empty_region(mlir_builder &bld, op_state &st) {
        auto reg = st.addRegion();
        reg->emplaceBlock();
        return reg;
    }

    hl::ValueYieldOp get_maybe_yield(region_t &reg) {
        return mlir::dyn_cast< hl::ValueYieldOp >(reg.back().back());
    }

    hl::ValueYieldOp get_yield(region_t &reg) {
        auto op = get_maybe_yield(reg);
        VAST_ASSERT(op && "getting yield from non-value region");
        return op;
    }

    mlir_value get_maybe_yielded_value(region_t &reg) {
        if (auto yield = get_maybe_yield(reg))
            return yield.getResult();
        return {};
    }

    mlir_value get_yielded_value(region_t &reg) {
        auto val = get_maybe_yielded_value(reg);
        VAST_ASSERT(val && "getting value from non-value region");
        return val;

    }

    mlir_type get_maybe_yielded_type(region_t &reg) {
        if (auto val = get_maybe_yielded_value(reg))
            return val.getType();
        return {};
    }

    mlir_type get_yielded_type(region_t &reg) {
        auto ty = get_maybe_yielded_type(reg);
        VAST_ASSERT(ty && "getting type from non-value region");
        return ty;
    }

    mlir::RegionSuccessor trivial_region_succ(region_t *reg) {
        return { reg, reg->getArguments() };
    }

} // namespace vast
