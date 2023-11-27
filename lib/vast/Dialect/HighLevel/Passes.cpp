// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/Passes.hpp"

namespace vast::cg {

    static pipeline_step_ptr splice_trailing_scopes() {
        return pass_pipeline_step::make(hl::createSpliceTrailingScopes);
    }

    pipeline_step_ptr make_canonicalize_pipeline() {
        return compound_pipeline_step::make({
            splice_trailing_scopes,
        });
    }

} // namespace vast::cg
