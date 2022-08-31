// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include <vast/Dialect/LowLevel/LowLevelDialect.hpp>
#include <memory>

namespace vast::ll
{
    std::unique_ptr< mlir::Pass > createToLLVMPass();

    /// Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Dialect/LowLevel/Passes.h.inc"

} // namespace vast::hl
