// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"
#include "vast/Dialect/Core/CoreDialect.hpp"

#include "vast/Dialect/HighLevel/Passes.hpp"

#include <memory>

namespace vast::hl
{
    // Generate the classes which represent the passes
    #define GEN_PASS_CLASSES
    #include "vast/Dialect/HighLevel/Passes.h.inc"

} // namespace vast::hl
