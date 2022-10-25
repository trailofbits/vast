// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

namespace vast
{
    #ifdef ENABLE_PDLL_CONVERSIONS
        constexpr bool enable_pdll_conversion_passes = true;

        std::unique_ptr< mlir::Pass > createHLToFuncPass();
    #endif

    // Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Conversion/Passes.h.inc"

} // namespace vast
