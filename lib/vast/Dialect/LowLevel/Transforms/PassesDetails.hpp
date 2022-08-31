// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"

#include <memory>

namespace vast::ll
{
    // Generate the classes which represent the passes
    #define GEN_PASS_CLASSES
    #include "vast/Dialect/LowLevel/Passes.h.inc"

} // namespace vast::ll
