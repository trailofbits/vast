// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/StringMap.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Pipeline.hpp"
#include "vast/Frontend/Options.hpp"
#include "vast/Frontend/Targets.hpp"

namespace vast::cc {

    struct pipelines_config {
        llvm::StringMap< pipeline_step_builder > pipelines;
    };

    pipelines_config default_pipelines_config();

    enum class pipeline_source { ast };

    //
    // Create pipeline schedule from source `src` to target `trg`
    //
    // Source can be either AST or MLIR dialect
    //
    // Target can be either MLIR dialect, LLVM IR or other downstream target
    // (object file, assembly, etc.)
    //
    // If the target is LLVM IR or other downstream target, the pipeline will
    // proceed into LLVM dialect.
    //
    // Pipeline configuration describe named pipelines that are used in the
    // scheduled pipeline.
    //
    std::unique_ptr< pipeline_t > setup_pipeline(
        pipeline_source src, output_type trg,
        mcontext_t &mctx,
        const vast_args &vargs,
        const pipelines_config &config
    );

} // namespace vast::cc

