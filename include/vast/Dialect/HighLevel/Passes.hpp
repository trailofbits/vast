// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/SCF/SCF.h>
VAST_UNRELAX_WARNINGS

#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <memory>

namespace vast::hl
{
    std::unique_ptr< mlir::Pass > createLowerHLToLLPass();

    std::unique_ptr< mlir::Pass > createLowerHighLevelTypesPass();

    std::unique_ptr< mlir::Pass > createStructsToTuplesPass();

    std::unique_ptr< mlir::Pass > createLowerHighLevelControlFlowPass();

    std::unique_ptr< mlir::Pass > createLLVMDumpPass();

    void registerHLToLLVMIR(mlir::DialectRegistry &);
    void registerHLToLLVMIR(mlir::MLIRContext &);

    /// Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Dialect/HighLevel/Passes.h.inc"

} // namespace vast::hl
