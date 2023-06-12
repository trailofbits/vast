// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"
#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/ABI/ABIDialect.hpp"

#include "vast/Conversion/Passes.hpp"

namespace vast
{
    // Generate the classes which represent the passes
    #define GEN_PASS_CLASSES
    #include "vast/Conversion/Passes.h.inc"

} // namespace vast
