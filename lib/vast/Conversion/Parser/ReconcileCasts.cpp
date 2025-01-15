#include "vast/Util/Warnings.hpp"

#include "vast/Conversion/Parser/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

namespace vast::conv {

    namespace pattern {

        using cast_conversions = util::type_list<
            // Casts
        >;

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
