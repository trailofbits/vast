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

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    for (auto region : op.getRegions()) {
                        if (!pr::is_noparse_region(region)) {
                            return true;
                        }
                    }

                    return false;
                });
            }
        };

        struct RefineReturn : operation_conversion_pattern< hl::ReturnOp >
        {
            using base = operation_conversion_pattern< hl::ReturnOp >;
            using base::base;

            using adaptor_t = typename hl::ReturnOp::Adaptor;

            logical_result matchAndRewrite(
                hl::ReturnOp op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto cast = adaptor.getResult().front().getDefiningOp();
                VAST_CHECK(mlir::isa< pr::Cast >(cast), "Expected cast op");
                rewriter.create< hl::ReturnOp >(op.getLoc(), cast->getOperand(0));

                auto cast_users = cast->getUsers();
                auto num_users  = std::distance(cast_users.begin(), cast_users.end());
                if (num_users == 1) {
                    rewriter.eraseOp(cast);
                }

                rewriter.eraseOp(op);
                return mlir::success();
            }

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< hl::ReturnOp >([](hl::ReturnOp op) {
                    if (pr::is_maybedata(op->getOperand(0))) {
                        auto result = op->getOperand(0).getDefiningOp();
                        if (auto cast = mlir::dyn_cast< pr::Cast >(result)) {
                            if (pr::is_nodata(cast.getOperand())) {
                                return false;
                            }
                        }
                    }

                    return true;
                });
            }
        };

        struct RefineDeclType : operation_conversion_pattern< pr::Decl >
        {
            using op_t = pr::Decl;
            using base = operation_conversion_pattern< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            static mlir_type resolve(mlir_type a, mlir_type b) {
                if (!a) {
                    return b;
                }
                return a == b ? a : pr::MaybeDataType::get(a.getContext());
            }

            static mlir_type assigned_type(mlir_value value) {
                if (!value.getDefiningOp()) {
                    return value.getType();
                } else if (auto cast = dyn_cast< pr::Cast >(value.getDefiningOp())) {
                    return assigned_type(cast.getOperand());
                } else {
                    return value.getType();
                }
            }

            static mlir_type assigned_type(op_t op) {
                mlir_type type = {};
                auto mod       = op->template getParentOfType< core::ModuleOp >();
                for (auto use : core::symbol_table::get_symbol_uses(op, mod)) {
                    if (auto ref = dyn_cast< pr::Ref >(use.getUser())) {
                        for (auto ref_user : ref->getUsers()) {
                            if (auto assign = dyn_cast< pr::Assign >(ref_user)) {
                                type = resolve(type, assigned_type(assign.getValue()));
                            } else if (isa< mlir::CallOpInterface >(ref_user)) {
                                return op.getType();
                            }
                        }
                    }
                }
                return op.getType();
            }

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                rewriter.modifyOpInPlace(op, [&] { op.setType(assigned_type(op)); });
                return mlir::success();
            }

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    return !pr::is_maybedata(op.getType())
                        || pr::is_maybedata(assigned_type(op));
                });
            }
        };

        template< typename op_t >
        struct DeadOpElimination : operation_conversion_pattern< op_t >
        {
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

        // clang-format off
        using refines = util::type_list<
            DeadOpElimination< pr::NoParse >,
            DeadOpElimination< pr::Cast >,
            NoParseFold< hl::IfOp >,
            NoParseFold< hl::WhileOp >,
            NoParseFold< hl::ForOp >,
            NoParseFold< hl::DoOp >,
            NoParseFold< hl::ChooseExprOp >,
            NoParseFold< hl::BinaryCondOp >,
            RefineReturn,
            RefineDeclType,
            DefinitionElimination< hl::EnumDeclOp >,
            DefinitionElimination< hl::StructDeclOp >,
            DefinitionElimination< hl::UnionDeclOp >,
            DefinitionElimination< hl::TypeDeclOp >
        >;
        // clang-format on

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
