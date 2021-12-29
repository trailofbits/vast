// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include <memory>

namespace vast::hl
{
    std::unique_ptr< mlir::Pass > createLowerHLToLLPass();

    std::unique_ptr< mlir::Pass > createLowerHighLevelTypesPass();

    std::unique_ptr< mlir::Pass > createStructsToTuplesPass();

    std::unique_ptr< mlir::Pass > createLowerHighLevelControlFlowPass();

    /// Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Dialect/HighLevel/Passes.h.inc"

} // namespace vast::hl
