// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
VAST_UNRELAX_WARNINGS

#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <memory>

namespace vast::hl
{
    std::unique_ptr< mlir::Pass > createHLToLLVMPass();

    std::unique_ptr< mlir::Pass > createHLLowerTypesPass();

    std::unique_ptr< mlir::Pass > createHLStructsToTuplesPass();

    std::unique_ptr< mlir::Pass > createHLStructsToLLVMPass();

    std::unique_ptr< mlir::Pass > createHLLowerEnumsPass();

    std::unique_ptr< mlir::Pass > createHLToSCFPass();

    std::unique_ptr< mlir::Pass > createLLVMDumpPass();

    std::unique_ptr< mlir::Pass > createExportFnInfoPass();

    std::unique_ptr< mlir::Pass > createHLToLLGEPsPass();

    void registerHLToLLVMIR(mlir::DialectRegistry &);
    void registerHLToLLVMIR(mlir::MLIRContext &);

    /// Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Dialect/HighLevel/Passes.h.inc"

} // namespace vast::hl
