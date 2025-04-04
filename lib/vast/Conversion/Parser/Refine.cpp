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

        template< typename op_t >
        struct NoParseFold : operation_conversion_pattern< op_t >
        {
            using base = operation_conversion_pattern< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                if (op->getNumResults() == 0) {
                    rewriter.eraseOp(op);
                    return mlir::success();
                } else {
                    rewriter.replaceOpWithNewOp< pr::NoParse >(
                        op, op->getResultTypes(), adaptor.getOperands()
                    );
                    return mlir::success();
                }

                return mlir::failure();
            }

            static bool is_nodata(mlir_type type) { return mlir::isa< pr::NoDataType >(type); }

            static bool is_nodata(mlir_value value) { return is_nodata(value.getType()); }

            static bool is_nodata(mlir::ValueRange values) {
                for (auto value : values) {
                    if (!is_nodata(value)) {
                        return false;
                    }
                }
                return true;
            }

            static bool is_noparse_op(mlir::Operation &op) {
                if (mlir::isa< pr::NoParse >(op)) {
                    return true;
                }

                if (mlir::isa< hl::NullStmt >(op)) {
                    return true;
                }

                if (mlir::isa< hl::BreakOp >(op)) {
                    return true;
                }

                if (mlir::isa< hl::ContinueOp >(op)) {
                    return true;
                }

                if (auto yield = mlir::dyn_cast< hl::CondYieldOp >(op)) {
                    if (is_nodata(yield.getResult())) {
                        return true;
                    }
                }

                if (auto yield = mlir::dyn_cast< hl::ValueYieldOp >(op)) {
                    if (is_nodata(yield.getResult())) {
                        return true;
                    }
                }

                if (auto ret = mlir::dyn_cast< hl::ReturnOp >(op)) {
                    if (is_nodata(ret.getResult())) {
                        return true;
                    }
                }

                if (auto call = mlir::dyn_cast< hl::CallOp >(op)) {
                    return is_nodata(call.getArgOperands()) && is_nodata(call.getResults());
                }

                return false;
            }

            static bool is_noparse_region(mlir::Region *region) {
                if (region->empty()) {
                    return true;
                }

                for (auto &block : *region) {
                    for (auto &op : block) {
                        if (!is_noparse_op(op)) {
                            return false;
                        }
                    }
                }

                return true;
            }

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    for (auto region : op.getRegions()) {
                        if (!is_noparse_region(region)) {
                            return true;
                        }
                    }

                    return false;
                });
            }
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
            NoParseFold< hl::IfOp >,
            NoParseFold< hl::WhileOp >,
            NoParseFold< hl::ForOp >,
            NoParseFold< hl::DoOp >,
            NoParseFold< hl::ChooseExprOp >,
            NoParseFold< hl::BinaryCondOp >,
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
