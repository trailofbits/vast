
#include "vast/Util/Warnings.hpp"

#include "vast/Conversion/Parser/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

namespace vast::conv
{
    struct HLToParserPass : ConversionPassMixin< HLToParserPass, HLToParserBase >
    {
        using base = ConversionPassMixin< HLToParserPass, HLToParserBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(auto &/* cfg */) {
        }
    };

} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createHLToParserPass() {
    return std::make_unique< vast::conv::HLToParserPass >();
}
