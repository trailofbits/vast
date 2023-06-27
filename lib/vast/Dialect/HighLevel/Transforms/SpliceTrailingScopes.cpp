// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Scopes.hpp"

#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{
    namespace
    {
        namespace pattern
        {
            using type_converter = mlir::TypeConverter;

            struct splice_trailing_scopes : generic_conversion_pattern
            {
                using base = generic_conversion_pattern;
                using base::base;

                splice_trailing_scopes(type_converter &tc, mcontext_t &mctx)
                    : base(tc, mctx) {}

                logical_result matchAndRewrite(
                        operation op,
                        mlir::ArrayRef< Value >,
                        conversion_rewriter &rewriter
                ) const override
                {
                    auto scope = mlir::dyn_cast< hl::ScopeOp >(op);
                    if(!scope)
                        return logical_result::failure();

                    auto &body = scope.getBody();
                    auto &start = body.front();
                    auto target = scope->getBlock();

                    rewriter.inlineRegionBefore(body, *op->getParentRegion(), target->getIterator());
                    rewriter.mergeBlocks(&start, target, mlir::ValueRange());
                    rewriter.eraseOp(op);

                    return logical_result::success();
                }
            };
        } // namespace pattern
    } // namespace

    struct SpliceTrailingScopes : ModuleConversionPassMixin< SpliceTrailingScopes, SpliceTrailingScopesBase >
    {
        using base = ModuleConversionPassMixin< SpliceTrailingScopes, SpliceTrailingScopesBase >;
        using config_t = typename base::config_t;

        static auto create_conversion_target(mcontext_t &mctx)
        {
            conversion_target target(mctx);

            auto is_legal = [](operation op)
            {
                return !is_trailing_scope(op);
            };

            target.markUnknownOpDynamicallyLegal(is_legal);
            return target;
        }

        void runOnOperation() override
        {
            auto &mctx = getContext();
            auto target = create_conversion_target(mctx);
            vast_module op = getOperation();

            rewrite_pattern_set patterns(&mctx);

            auto tc = pattern::type_converter();
            patterns.template add< pattern::splice_trailing_scopes >(tc, mctx);

            if (mlir::failed(mlir::applyPartialConversion(op,
                                                          target,
                                                          std::move(patterns))))
            {
                return signalPassFailure();
            }
        }

    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createSpliceTrailingScopes()
{
    return std::make_unique< vast::hl::SpliceTrailingScopes >();
}
