// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"

namespace vast::conv::pipeline {

    static pipeline_step_ptr llvm_debug_scope() {
        // This is necessary to have line tables emitted and basic debugger
        // working. In the future we will add proper debug information emission
        // directly from our frontend.
        return pass(mlir::LLVM::createDIScopeForLLVMFuncOpPass)
            .depends_on(core_to_llvm);
    }

    pipeline_step_ptr core_to_llvm() {
        // TODO add dependencies
        return pass(createCoreToLLVMPass);
    }

    pipeline_step_ptr to_llvm() {
        return compose(irs_to_llvm, core_to_llvm, llvm_debug_scope);
    }

} // namespace vast::conv::pipeline
