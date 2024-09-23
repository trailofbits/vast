// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Pipeline.hpp"

#include "vast/Conversion/Passes.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"


namespace vast::hl::pipeline {

    pipeline_step_ptr splice_trailing_scopes() {
        return pass(hl::createSpliceTrailingScopes);
    }

    //
    // desugar pipeline passes
    //
    static pipeline_step_ptr lower_typedefs() {
        return pass(hl::createLowerTypeDefsPass);
    }

    static pipeline_step_ptr lower_elaborated_types() {
        return pass(hl::createLowerElaboratedTypesPass);
    }

    // TODO: add more passes here (remove elaborations, decayed types, lvalue types etc.)
    pipeline_step_ptr desugar() {
        return compose("desugar", lower_typedefs).depends_on(lower_elaborated_types);
    }

    //
    // simplifcaiton passes
    //
    static pipeline_step_ptr dce() {
        return pass(hl::createDCEPass).depends_on(splice_trailing_scopes);
    }

    static pipeline_step_ptr ude() {
        return pass(hl::createUDEPass).depends_on(splice_trailing_scopes);
    }

    static pipeline_step_ptr lower_enum_refs() {
        return pass(hl::createLowerEnumRefsPass).depends_on(desugar);
    }

    static pipeline_step_ptr lower_enums() {
        return compose("lower_enums", lower_enum_refs);
    }

    pipeline_step_ptr simplify() {
        return compose("simplify",
            conv::pipeline::to_hlbi,
            ude,
            dce,
            desugar
        );
    }

    //
    // stdtypes passes
    //
    pipeline_step_ptr stdtypes() {
        return pass(hl::createHLLowerTypesPass).depends_on(desugar, lower_enums);
    }

} // namespace vast::hl::pipeline
