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

    enum class pipeline_source { ast };

    struct vast_pipeline : pipeline_t
    {
        using base = pipeline_t;

        vast_pipeline(mcontext_t &mctx, const vast_args &vargs)
            : base(&mctx), vargs(vargs)
        {}

        virtual ~vast_pipeline() = default;

        void schedule(pipeline_step_ptr step) override;

        bool is_disabled(const pipeline_step_ptr &step) const;

        const vast_args &vargs;
    };


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
    std::unique_ptr< vast_pipeline > setup_pipeline(
        pipeline_source src, target_dialect trg,
        mcontext_t &mctx,
        const vast_args &vargs
    );

} // namespace vast::cc
