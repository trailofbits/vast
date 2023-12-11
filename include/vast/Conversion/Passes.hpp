// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>
#include <vast/Dialect/LowLevel/LowLevelDialect.hpp>
#include <vast/Dialect/Core/CoreDialect.hpp>
#include <vast/Dialect/ABI/ABIDialect.hpp>

#include <vast/Util/Pipeline.hpp>

#include <memory>

namespace vast
{
    #ifdef ENABLE_PDLL_CONVERSIONS
        constexpr bool enable_pdll_conversion_passes = true;

        namespace pdll
        {
            std::unique_ptr< mlir::Pass > createHLToFuncPass();
        } // namespace pdll
    #endif

    // Common
    std::unique_ptr< mlir::Pass > createIRsToLLVMPass();

    // Core
    std::unique_ptr< mlir::Pass > createCoreToLLVMPass();

    // ABI
    std::unique_ptr< mlir::Pass > createEmitABIPass();

    std::unique_ptr< mlir::Pass > createLowerABIPass();

    // FromHL
    std::unique_ptr< mlir::Pass > createHLToLLCFPass();

    std::unique_ptr< mlir::Pass > createHLToLLGEPsPass();

    std::unique_ptr< mlir::Pass > createHLToLLVarsPass();

    std::unique_ptr< mlir::Pass > createHLEmitLazyRegionsPass();

    std::unique_ptr< mlir::Pass > createHLToLLFuncPass();

    // Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Conversion/Passes.h.inc"

    // TODO(conv): Define dependencies between these.
    static inline void build_abi_pipeline(mlir::PassManager &pm)
    {
        pm.addPass(createEmitABIPass());
        pm.addPass(createLowerABIPass());
    }

    static inline void build_to_ll_pipeline(mlir::PassManager &pm)
    {
        pm.addPass(createHLToLLFuncPass());
        pm.addPass(createHLToLLVarsPass());
        pm.addPass(createHLToLLCFPass());
        pm.addPass(createHLEmitLazyRegionsPass());
        pm.addPass(createHLToLLGEPsPass());
    }

    static inline void build_to_llvm_pipeline(mlir::PassManager &pm)
    {
        pm.addPass(createIRsToLLVMPass());
        pm.addPass(createCoreToLLVMPass());
    }

    namespace conv::pipeline
    {
        pipeline_step_ptr abi();
        pipeline_step_ptr irs_to_llvm();
        pipeline_step_ptr core_to_llvm();

        pipeline_step_ptr to_ll();

        pipeline_step_ptr to_llvm();

    } // namespace conv::pipeline

} // namespace vast
