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

#include "vast/Util/Terminator.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

namespace vast::conv {

    namespace pattern {

        struct EmptyDefaultOpElimination : operation_conversion_pattern< hl::DefaultOp >
        {
            using op_t = hl::DefaultOp;
            using base = operation_conversion_pattern< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            static bool is_empty(op_t op) {
                if (op.getBody().empty())
                    return true;
                return op.getBody().front().empty();
            }

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                rewriter.eraseOp(op);
                return mlir::success();
            }

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    return !is_empty(op);
                });
            }
        };

        // clang-format off
        using refines = util::type_list<
            EmptyDefaultOpElimination
        >;
        // clang-format on

    } // namespace pattern

    struct RefineCleanUpPass : ConversionPassMixin< RefineCleanUpPass, ParserRefineCleanUpBase >
    {
        using base = ConversionPassMixin< RefineCleanUpPass, ParserRefineCleanUpBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::refines >(cfg);
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createParserRefineCleanUpPass() {
    return std::make_unique< vast::conv::RefineCleanUpPass >();
}
