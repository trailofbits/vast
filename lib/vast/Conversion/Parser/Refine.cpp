#include "vast/Util/Warnings.hpp"

#include "vast/Conversion/Parser/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"
#include "Utils.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

namespace vast::conv {

    namespace pattern {

        template< typename op_t >
        struct refine_covnersion_pattern_base
            : mlir_pattern_mixin< operation_conversion_pattern< op_t > >
            , mlir::OpConversionPattern< op_t >
        {
            using base = mlir::OpConversionPattern< op_t >;
            using base::base;
        };

        struct DeclRefine
            : refine_covnersion_pattern_base< pr::Decl >
        {
            using op_t = pr::Decl;
            using base = refine_covnersion_pattern_base< pr::Decl>;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                return mlir::success();
            }
        };

        using refines = util::type_list< DeclRefine >;

    } // namespace pattern

    struct ParserRefinePass
        : ConversionPassMixin< ParserRefinePass, ParserRefineBase >
    {
        using base = ConversionPassMixin< ParserRefinePass, ParserRefineBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::refines >(cfg);
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createParserRefinePass() {
    return std::make_unique< vast::conv::ParserRefinePass >();
}
