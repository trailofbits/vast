// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Support/LogicalResult.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/Common.hpp"

namespace vast {

    region_t* build_region(mlir_builder &bld, op_state &st, builder_callback_ref callback);
    region_t* build_region(mlir_builder &bld, op_state &st, maybe_builder_callback_ref callback);
    region_t* build_region(region_t *reg, mlir_builder &bld, op_state &st, builder_callback_ref callback);
    region_t* build_region(region_t *reg, mlir_builder &bld, op_state &st, maybe_builder_callback_ref callback);

    region_t* build_empty_region(mlir_builder &bld, op_state &st);

    hl::ValueYieldOp get_maybe_yield(region_t &reg);
    hl::ValueYieldOp get_yield(region_t &reg);

    mlir_value get_maybe_yielded_value(region_t &reg);
    mlir_value get_yielded_value(region_t &reg);

    mlir_type get_maybe_yielded_type(region_t &reg);
    mlir_type get_yielded_type(region_t &reg);

    mlir::RegionSuccessor trivial_region_succ(region_t *reg);

} // namespace vast
