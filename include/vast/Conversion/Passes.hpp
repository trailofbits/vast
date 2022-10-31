// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>
#include <vast/Dialect/LowLevel/LowLevelDialect.hpp>

#include <memory>

namespace vast
{
    #ifdef ENABLE_PDLL_CONVERSIONS
        constexpr bool enable_pdll_conversion_passes = true;

        std::unique_ptr< mlir::Pass > createHLToFuncPass();
    #endif

    std::unique_ptr< mlir::Pass > createCoreToLLVMPass();

    std::unique_ptr< mlir::Pass > createHLFuncToFuncPass();

    // Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Conversion/Passes.h.inc"

} // namespace vast
