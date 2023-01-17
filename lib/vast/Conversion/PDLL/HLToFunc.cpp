// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"
#include "vast/Util/Common.hpp"

#include "../PassesDetails.hpp"

#include "vast/Conversion/HLToFunc.hpp"

namespace vast::pdll
{

    using RewritePatternSet = mlir::RewritePatternSet;
    using FrozenRewritePatternSet = mlir::FrozenRewritePatternSet;

    struct HLToFuncPass : HLToFuncBase< HLToFuncPass >
    {
        using HLToFuncBase::HLToFuncBase;

        LogicalResult initialize(mcontext_t *ctx) override {
            // Build the pattern set within the `initialize` to avoid recompiling PDL
            // patterns during each `runOnOperation` invocation.
            RewritePatternSet pattern_list(ctx);
            populateGeneratedPDLLPatterns(pattern_list);
            patterns = std::move(pattern_list);
            return mlir::success();
        }

        void runOnOperation() override {
            if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patterns))) {
                return signalPassFailure();
            }
        }

        FrozenRewritePatternSet patterns;
    };

    std::unique_ptr< mlir::Pass > createHLToFuncPass()
    {
        return std::make_unique< HLToFuncPass >();
    }

} // namespace vast::pdll
