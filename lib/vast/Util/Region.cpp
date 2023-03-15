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

    hl::ValueYieldOp get_maybe_yielded_value(Region &reg) {
        return mlir::dyn_cast< hl::ValueYieldOp >(reg.back().back());
    }

    hl::ValueYieldOp get_yielded_value(Region &reg) {
        auto op = get_maybe_yielded_value(reg);
        VAST_ASSERT(op && "getting value from non-value region");
        return op;
    }

    mlir_type get_yielded_type(Region &reg) {
        return get_yielded_value(reg).getResult().getType();
    }

    mlir::RegionSuccessor trivial_region_succ(Region *reg) {
        return { reg, reg->getArguments() };
    }

} // namespace vast
