// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"

namespace vast::conv::pipeline {

    static pipeline_step_ptr emit_abi() {
        // TODO add dependencies
        return pass(createEmitABIPass);
    }

    static pipeline_step_ptr lower_abi() {
        return pass(createLowerABIPass).depends_on(emit_abi);
    }

    pipeline_step_ptr abi() {
        return lower_abi();
    }

} // namespace vast::conv::pipeline
