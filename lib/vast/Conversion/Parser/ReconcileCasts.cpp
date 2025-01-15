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

        struct UnrealizedCastConversion
            : one_to_one_conversion_pattern< mlir::UnrealizedConversionCastOp, pr::Cast >
        {
            using op_t = mlir::UnrealizedConversionCastOp;
            using base = one_to_one_conversion_pattern< mlir::UnrealizedConversionCastOp, pr::Cast >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                if (op.getNumOperands() != 1) {
                    return mlir::failure();
                }

                auto src = mlir::dyn_cast< mlir::UnrealizedConversionCastOp >(op.getOperand(0).getDefiningOp());

                if (!src || src.getNumOperands() != 1) {
                    return mlir::failure();
                }

                if (pr::is_parser_type(src.getOperand(0).getType())) {
                    rewriter.replaceOpWithNewOp< pr::Cast >(op, op.getType(0), src.getOperand(0));
                    return mlir::success();
                }

                return mlir::success();
            }

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addLegalOp< pr::Cast >();
            }
        };

        using cast_conversions = util::type_list< UnrealizedCastConversion >;

    } // namespace pattern

    struct ParserReconcileCastsPass
        : ConversionPassMixin< ParserReconcileCastsPass, ParserReconcileCastsBase >
    {
        using base = ConversionPassMixin< ParserReconcileCastsPass, ParserReconcileCastsBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::cast_conversions >(cfg);
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createParserReconcileCastsPass() {
    return std::make_unique< vast::conv::ParserReconcileCastsPass >();
}
