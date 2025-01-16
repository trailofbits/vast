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

#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

namespace vast::conv {

    // FIXME simplify to required components
    struct function_type_converter
        : tc::identity_type_converter
        , tc::mixins< function_type_converter >
    {
        mcontext_t &mctx;
        std::vector< mlir_type > param_types;

        using mixin_base = tc::mixins< function_type_converter >;
        using mixin_base::convert_type_to_type;

        explicit function_type_converter(mcontext_t &mctx, std::vector< mlir_type > param_types)
            : mctx(mctx), param_types(std::move(param_types))
        {
            addConversion([this](tc::core_function_type ty) -> maybe_type_t {
                llvm::errs() << "Convert function type\n";
                return convert_type_to_type(ty);
            });
        }

        maybe_type_t convert_arg_type(mlir_type ty, unsigned long idx) const {
            llvm::errs() << "Convert arg type: " << idx << "\n";
            ty.dump();
            param_types[idx].dump();
            return param_types[idx];
        }

        maybe_types_t convert_types_to_types(auto types) const {
            types_t out;

            for (auto t : types) {
                if (auto ty = convert_type_to_type(t)) {
                    out.push_back(ty.value());
                } else {
                    return {};
                }
            }

            return { out };
        }

        maybe_type_t convert_type_to_type(tc::core_function_type ty) const {
            auto sig = signature_conversion(ty.getInputs());
            if (!sig) {
                return std::nullopt;
            }

            auto rty = convert_types_to_types(ty.getResults());
            if (!rty) {
                return std::nullopt;
            }

            return tc::core_function_type::get(
                ty.getContext(), sig->getConvertedTypes(), *rty, ty.isVarArg()
            );
        }

        maybe_type_t convert_type_to_type(mlir_type ty) const {
            if (auto ft = mlir::dyn_cast< tc::core_function_type >(ty)) {
                return convert_type_to_type(ft);
            }
            return ty;
        }
    };

    namespace pattern {

        template< typename op_t >
        struct refine_covnersion_pattern_base
            : mlir_pattern_mixin< operation_conversion_pattern< op_t > >
            , mlir::OpConversionPattern< op_t >
        {
            using base = mlir::OpConversionPattern< op_t >;
            using base::base;
        };

        struct FunctionRefine
            : refine_covnersion_pattern_base< hl::FuncOp >
            , tc::op_type_conversion< hl::FuncOp, function_type_converter >
        {
            using op_t = hl::FuncOp;
            using base = refine_covnersion_pattern_base< hl::FuncOp >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            static gap::generator< operation > calls_of(hl::FuncOp fn) {
                // FIXME: use get_effective_symbol_table_for
                auto scope = fn->getParentOfType< core::ModuleOp >();
                std::vector< operation > calls;
                // FIXME: use core::symbol_table::get_symbol_uses
                scope->walk([&](operation op) {
                    if (auto call = mlir::dyn_cast< hl::CallOp >(op)) {
                        if (call.getCallee() == fn.getSymName()) {
                            calls.push_back(call);
                        }
                    }
                });

                for (auto call : calls) {
                    co_yield call;
                }
            }

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto fty = op.getFunctionType();
                std::vector< mlir_type > param_types = fty.getInputs();
                bool change = false;

                for (auto call : calls_of(op)) {
                    for (auto [idx, operand] : llvm::enumerate(call->getOperands())) {
                        auto met = pr::meet(param_types[idx], operand.getType());
                        change = change || (met && (met != param_types[idx]));
                        param_types[idx] = met;
                    }
                }

                if (change) {
                    std::vector< mlir_type > new_param_types;
                    auto old_param_types = fty.getInputs();
                    for (auto [idx, ty] : llvm::enumerate(param_types)) {
                        new_param_types.push_back(ty ? ty : old_param_types[idx]);
                    }

                    auto new_fty = core::FunctionType::get(
                        rewriter.getContext(), new_param_types, fty.getResults(), fty.isVarArg()
                    );

                    rewriter.modifyOpInPlace(op, [&] () {
                        op.setFunctionType(new_fty);

                        auto &region = op.getBody();
                        if (region.empty())
                            return;

                        auto entry_block = &region.front();
                        for (auto [idx, arg] : llvm::enumerate(entry_block->getArguments())) {
                            arg.setType(new_param_types[idx]);
                        }
                    });

                    return mlir::success();
                }

                return mlir::failure();
            }

            static void legalize(base_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    // FIXME: duplicates param type resolution
                    auto fty = op.getFunctionType();
                    auto param_types = fty.getInputs();

                    for (auto call : calls_of(op)) {
                        for (auto [idx, operand] : llvm::enumerate(call->getOperands())) {
                            auto met = pr::meet(param_types[idx], operand.getType());
                            if (met && (met != param_types[idx])) {
                                return false;
                            }
                        }
                    }

                    return true;
                });
            }
        };

        struct DeclRefine
            : refine_covnersion_pattern_base< pr::Decl >
        {
            using op_t = pr::Decl;
            using base = refine_covnersion_pattern_base< pr::Decl>;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            gap::generator< pr::Assign > assigns_to(operation symbol) const {
                // FIXME: use get_effective_symbol_table_for
                auto scope = symbol->getParentOfType< core::ModuleOp >();
                for (auto use : core::symbol_table::get_symbol_uses(symbol, scope)) {
                    auto ref = mlir::dyn_cast< pr::Ref >(use.getUser());
                    VAST_ASSERT(ref);
                    for (auto user : ref->getUsers()) {
                        if (auto assign = mlir::dyn_cast< pr::Assign >(user)) {
                            if (assign.getTarget() == ref) {
                                co_yield assign;
                            }
                        }
                    }
                }
            }

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                for (auto assign : assigns_to(op)) {
                    assign->dump();
                }

                return mlir::failure();
            }
        };

        using refines = util::type_list< FunctionRefine >;

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
