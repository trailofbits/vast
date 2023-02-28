// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/BlockAndValueMapping.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"
#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Util/TypeList.hpp"
#include "vast/Util/DialectConversion.hpp"

namespace vast::hl
{
    using conversion_rewriter = mlir::ConversionPatternRewriter;

    template< typename source, typename LOp >
    struct bin_lop_pattern : operation_conversion_pattern< source > {
        using base = operation_conversion_pattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;

        auto region_value_yield_type(Region &region) const {
            return region.back().back().getOperand(0).getType();
        }

        logical_result matchAndRewrite(
            source op, adaptor_t adaptor, conversion_rewriter &rewriter) const override
        {
            auto lazy_side = [&] (Region &side) {
                auto lazy_op = rewriter.create< core::LazyOp >(
                    op.getLoc(), region_value_yield_type(side)
                );
                auto &lazy_region = lazy_op.getLazy();
                rewriter.inlineRegionBefore(side, lazy_region, lazy_region.end());
                return lazy_op;
            };

            auto lhs_lazy = lazy_side(op.getLhs());
            auto rhs_lazy = lazy_side(op.getRhs());

            auto logical = rewriter.create< LOp >(op.getLoc(), op.getType(), lhs_lazy, rhs_lazy);
            rewriter.replaceOp(op, { logical });
            return logical_result::success();
        }

        static void legalize(conversion_target &target) {
            target.addLegalOp< core::BinLAndOp, core::BinLOrOp >();
            target.addIllegalOp< hl::BinLAndOp, hl::BinLOrOp >();
        }
    };

    using bin_lop_conversions = util::type_list<
        bin_lop_pattern< hl::BinLAndOp, core::BinLAndOp >,
        bin_lop_pattern< hl::BinLOrOp,  core::BinLOrOp >
    >;

    struct HLEmitLazyRegionsPass
        : ModuleConversionPassMixin< HLEmitLazyRegionsPass, HLEmitLazyRegionsBase >
    {

        using base = ModuleConversionPassMixin< HLEmitLazyRegionsPass, HLEmitLazyRegionsBase>;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);
            target.addLegalDialect< vast::core::CoreDialect >();
            return target;
        }

        static void populate_conversions(rewrite_pattern_set &patterns) {
            populate_conversions_base<
                bin_lop_conversions
            >(patterns);
        }
    };

    std::unique_ptr< mlir::Pass > createHLEmitLazyRegionsPass() {
        return std::make_unique< HLEmitLazyRegionsPass >();
    }

} // namespace vast::hl
