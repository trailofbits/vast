// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"
#include "vast/Conversion/Common/Block.hpp"
#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Util/TypeList.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Terminator.hpp"
#include "vast/Util/Region.hpp"

namespace vast
{
    using conversion_rewriter = mlir::ConversionPatternRewriter;

    struct lazy_utils
    {
        static auto lazy_side(auto &&rewriter, mlir::Location loc, mlir::Region &side)
        {
            // Passed by val, we are expecting only mlir lightweight objects.
            auto mk_lazy_op = [&](auto ... args)
            {
                return rewriter.template create< core::LazyOp >( loc, args ... );
            };

            auto lazy_op = [&]
            {
                if (!terminator_t< hl::ValueYieldOp >::get(side.front()))
                {
                    return mk_lazy_op(mlir::NoneType::get(rewriter.getContext()));
                }
                return mk_lazy_op(get_yielded_type(side));
            }();

            auto &lazy_region = lazy_op.getLazy();
            rewriter.inlineRegionBefore(side, lazy_region, lazy_region.end());
            return lazy_op;
        }
    };

    template< typename source, typename LOp >
    struct bin_lop_pattern : operation_conversion_pattern< source >,
                             lazy_utils
    {
        using base = operation_conversion_pattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;

        logical_result matchAndRewrite(
            source op, adaptor_t adaptor, conversion_rewriter &rewriter) const override
        {
            auto lhs_lazy = lazy_side(rewriter, op.getLoc(), op.getLhs());
            auto rhs_lazy = lazy_side(rewriter, op.getLoc(), op.getRhs());

            auto logical = rewriter.create< LOp >(op.getLoc(), op.getType(),
                                                  lhs_lazy, rhs_lazy);
            rewriter.replaceOp(op, { logical });
            return logical_result::success();
        }

        static void legalize(conversion_target &target) {
            target.addLegalOp< LOp >();
            target.addIllegalOp< source >();
        }
    };

    struct cond_op : operation_conversion_pattern< hl::CondOp >,
                     lazy_utils
    {
        using source = hl::CondOp;
        using base = operation_conversion_pattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;

        logical_result matchAndRewrite(source op, adaptor_t adaptor,
                                       conversion_rewriter &rewriter) const override
        {
            auto &cond_block = op.getCondRegion().front();
            VAST_PATTERN_CHECK(conv::size(op.getCondRegion()) == 1,
                               "Unsupported shape of cond region of hl::CondOp:\n{0}", op);

            auto yield = terminator_t< hl::CondYieldOp >::get(cond_block);
            VAST_PATTERN_CHECK(yield, "Was not able to retrieve cond yield, {0}.", op);

            rewriter.mergeBlockBefore(&cond_block, op, std::nullopt);

            auto then_region = lazy_side(rewriter, op.getLoc(), op.getThenRegion());
            auto else_region = lazy_side(rewriter, op.getLoc(), op.getElseRegion());

            auto yielded_val = yield.op().getResult();
            auto select = rewriter.create< core::SelectOp >(
                    op.getLoc(),
                    then_region.getType(), yielded_val,
                    then_region, else_region);

            rewriter.eraseOp(yield.op());
            rewriter.replaceOp(op, select.getResults());
            return mlir::success();
        }

        static void legalize(conversion_target &target) {
            target.addLegalOp< core::SelectOp >();
            target.addIllegalOp< hl::CondOp >();
        }
    };

    using bin_lop_conversions = util::type_list<
        bin_lop_pattern< hl::BinLAndOp, core::BinLAndOp >,
        bin_lop_pattern< hl::BinLOrOp,  core::BinLOrOp >,
        cond_op
    >;

    struct HLEmitLazyRegionsPass
        : ModuleConversionPassMixin< HLEmitLazyRegionsPass, HLEmitLazyRegionsBase >
    {
        using base = ModuleConversionPassMixin< HLEmitLazyRegionsPass, HLEmitLazyRegionsBase>;
        using config_t = typename base::config_t;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);
            target.addLegalDialect< vast::core::CoreDialect >();
            return target;
        }

        static void populate_conversions(config_t &config) {
            populate_conversions_base<
                bin_lop_conversions
            >(config);
        }
    };

    std::unique_ptr< mlir::Pass > createHLEmitLazyRegionsPass() {
        return std::make_unique< HLEmitLazyRegionsPass >();
    }

} // namespace vast
