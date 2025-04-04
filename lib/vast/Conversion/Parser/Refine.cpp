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
        struct DefinitionElimination : erase_pattern< op_t >
        {
            using base = erase_pattern< op_t >;
            using base::base;
        };

        struct DeadNoParseElimination : operation_conversion_pattern< pr::NoParse >
        {
            using op_t = pr::NoParse;
            using base = operation_conversion_pattern< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                if (op->getUsers().empty()) {
                    rewriter.eraseOp(op);
                    return mlir::success();
                }

                return mlir::failure();
            }

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    return !op->getUsers().empty();
                });
            }
        };

        using refines = util::type_list<
            DeadNoParseElimination,
            DefinitionElimination< hl::EnumDeclOp >,
            DefinitionElimination< hl::StructDeclOp >,
            DefinitionElimination< hl::UnionDeclOp >,
            DefinitionElimination< hl::TypeDeclOp >
        >;

    } // namespace pattern

    struct ParserRefinePass : ConversionPassMixin< ParserRefinePass, ParserRefineBase >
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
