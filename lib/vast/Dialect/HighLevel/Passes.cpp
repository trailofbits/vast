// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/Passes.hpp"

namespace vast::hl::pipeline {

    //
    // canonicalization pipeline passes
    //
    static pipeline_step_ptr splice_trailing_scopes() {
        return pass(hl::createSpliceTrailingScopes);
    }

    // TODO: add more passes here (remove reduntant skips etc.)

    pipeline_step_ptr canonicalize() {
        return compose(splice_trailing_scopes);
    }

    //
    // desugar pipeline passes
    //
    static pipeline_step_ptr lower_types() {
        return pass(hl::createLowerTypeDefsPass);
    }

    // TODO: add more passes here (remove elaborations, decayed types, lvalue types etc.)

    pipeline_step_ptr desugar() {
        return compose(lower_types);
    }

    //
    // simplifcaiton passes
    //
    static pipeline_step_ptr dce() {
        return pass(hl::createDCEPass).depends_on(canonicalize);
    }

    pipeline_step_ptr simplify() {
        return compose(optional< dce >, optional< desugar >);
    }

    //
    // stdtypes passes
    //
    static pipeline_step_ptr lower_types_to_std() {
        return pass(hl::createHLLowerTypesPass);
    }

    pipeline_step_ptr stdtypes() {
        return compose(lower_types_to_std).depends_on(desugar);
    }

} // namespace vast::hl::pipeline
