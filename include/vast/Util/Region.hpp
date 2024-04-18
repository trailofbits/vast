// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Support/LogicalResult.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/Common.hpp"

namespace vast {

    Region* fill_region(Region *reg, Builder &bld, State &st, BuilderCallback callback);

    Region* build_region(Builder &bld, State &st, BuilderCallback callback);
    Region* build_empty_region(Builder &bld, State &st);

    hl::ValueYieldOp get_maybe_yield(Region &reg);
    hl::ValueYieldOp get_yield(Region &reg);

    mlir_value get_maybe_yielded_value(Region &reg);
    mlir_value get_yielded_value(Region &reg);

    mlir_type get_maybe_yielded_type(Region &reg);
    mlir_type get_yielded_type(Region &reg);

    mlir::RegionSuccessor trivial_region_succ(Region *reg);

} // namespace vast
