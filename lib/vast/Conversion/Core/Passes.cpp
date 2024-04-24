// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"

namespace vast::conv::pipeline {

    static pipeline_step_ptr llvm_debug_scope() {
        // This is necessary to have line tables emitted and basic debugger
        // working. In the future we will add proper debug information emission
        // directly from our frontend.
        return nested< vast_module >(mlir::LLVM::createDIScopeForLLVMFuncOpPass)
            .depends_on(core_to_llvm);
    }

    pipeline_step_ptr core_to_llvm() {
        // TODO add dependencies
        return pass(createCoreToLLVMPass)
            .depends_on(to_ll, irs_to_llvm);
    }

    pipeline_step_ptr to_llvm() {
        return compose("to-llvm", irs_to_llvm, core_to_llvm, llvm_debug_scope);
    }


    pipeline_step_ptr canonicalize() {
        return pass([] {
            return mlir::createCanonicalizerPass();
        });
    }

} // namespace vast::conv::pipeline
