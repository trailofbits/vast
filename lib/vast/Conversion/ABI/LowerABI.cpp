// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include <mlir/Rewrite/PatternApplicator.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/ABI/ABI.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/TypeSwitch.hpp"
#include "vast/Util/TypeUtils.hpp"

#include "vast/Dialect/ABI/ABIOps.hpp"

#include "vast/Conversion/Common/Passes.hpp"

#include <iostream>
#include <unordered_map>

#include <gap/core/generator.hpp>

#include "vast/Conversion/ABI/AggregateTypes.hpp"

namespace vast
{
    using values        = gap::generator< mlir::Value >;
    using frozen_values = std::vector< mlir::Value >;

    namespace pattern
    {
    namespace
    {
        bool can_concat_as(const auto &dl, auto type_range, auto target)
        {
            return bw(dl, target) == bw(dl, type_range);
        }

        // [ `current_arg`, offset into `current_arg`, size of `current_arg` ]
        using arg_list_position = std::tuple< std::size_t, std::size_t, std::size_t >;

        template< typename Op >
        struct abi_pattern_base : operation_conversion_pattern< Op >
        {
            using base = operation_conversion_pattern< Op >;
            using op_t = Op;

            const mlir::DataLayout &dl;

            template< typename ... Args >
            abi_pattern_base(const mlir::DataLayout &dl, Args && ... args)
                : base(std::forward< Args >(args) ...),
                  dl(dl)
            {}

            using state_capture = match_and_rewrite_state_capture< op_t >;

            static void legalize(conversion_target &trg)
            {
                trg.addIllegalOp< op_t >();
            }

            // Simplifies implementation.
            virtual values match_on(abi::DirectOp direct, state_capture &state) const = 0;

            virtual values match_on(mlir::Operation *op, state_capture &state) const
            {
                co_return;
            }

            values dispatch(mlir::Operation *op, state_capture state) const
            {
                if (auto direct = mlir::dyn_cast< abi::DirectOp >(op))
                    return this->match_on(direct, state);
                return this->match_on(op, state);
            }

            static auto get(auto &&thing, long idx)
            {
                VAST_ASSERT(std::distance(thing.begin(), thing.end()) > idx);
                return *std::next(thing.begin(), idx);
            }

            auto bw(mlir_type type) const
            {
                return ::vast::bw(dl, hl::strip_value_category(type));
            }
            auto bw(mlir::Value val) const { return bw(val.getType()); }

        };

        template< typename Op >
        struct function_border_base : abi_pattern_base< Op >
        {
            using base = abi_pattern_base< Op >;
            using op_t = typename  base::op_t;
            using state_capture = typename base::state_capture;

            using base::base;


            logical_result rewrite(state_capture state) const
            {
                std::vector< mlir::Value > to_replace;
                for (auto &nested : state.op.getBody().getOps())
                {
                    for (auto converted : base::dispatch(&nested, state))
                        to_replace.push_back(converted);

                }

                state.rewriter.replaceOp(state.op, to_replace);
                return mlir::success();
            }
        };

        // TODO(conv:abi): If they end up being the same, rename the `function_border_base`
        //                 and erase the alias.
        template< typename op_t >
        using call_border = function_border_base< op_t >;

        template< typename op_t >
        using state_capture = match_and_rewrite_state_capture< op_t >;

        template< typename state_t, typename pattern_t >
        struct deconstructs_types
        {
            state_t &state;
            pattern_t &pattern;

            deconstructs_types(state_t &state, pattern_t &pattern)
                : state(state), pattern(pattern)
            {}

            /* abi::DirectOp related methods. */

            // It is important that this cannot be called on rvalue, due
            // to lifeties and coroutines.
            values handle(abi::DirectOp direct) &
            {
                // TODO(conv:abi): Can direct have more return types?
                VAST_ASSERT(direct.getNumResults() >= 1 && direct.getNumOperands() == 1);

                std::vector< mlir::Value > init_values;
                for (auto res_type : direct.getOperands().getTypes())
                {
                    auto converted = convert(res_type, direct);
                    init_values.insert(init_values.end(), converted.begin(),
                                                          converted.end());
                }

                VAST_CHECK(init_values.size() == direct.getNumResults(),
                           "{0} != {1}", init_values.size(), direct.getNumResults());

                for (std::size_t i = 0; i < init_values.size(); ++i)
                {
                    auto var = state.rewriter.template create< ll::UninitializedVar >(
                            direct.getLoc(), direct.getResults()[i].getType());

                    co_yield state.rewriter.template create< ll::InitializeVar >(
                            direct.getLoc(),
                            var.getResult().getType(),
                            var, init_values[i]);
                }
            }


            auto deconstruct_record(hl::RecordType record_type, abi::DirectOp direct)
            {
                auto val = direct.getOperand(0).getDefiningOp();
                return conv::abi::deconstruct_aggregate(pattern, direct, val, state.rewriter);
            }

            mlir::Value convert_primitive_type(mlir_type target_type, abi::DirectOp direct)
            {
                VAST_CHECK(direct.getNumOperands() >= 1,
                           "abi.direct op should have > 1 operands: {0}", direct);

                // We need to reconstruct the type and we ?know? that arguments
                // simply need to be concated?
                VAST_CHECK(can_concat_as(pattern.dl, direct.getOperands().getTypes(),
                                         target_type),
                           "Cannot do concat when converting {0}", direct);

                return state.rewriter.template create< ll::Concat >(
                        direct.getLoc(),
                        target_type, direct.getOperands());
            }

            auto convert(mlir::Type target_type, abi::DirectOp direct)
                -> std::vector< mlir::Value >
            {
                // In epilogue, the type category is ?not? an lvalue.
                if (target_type == direct.getResult()[0].getType())
                    return { direct.getOperand(0) };


                auto naked = hl::strip_elaborated(target_type);
                if (auto record_type = mlir::dyn_cast< hl::RecordType >(naked))
                    return deconstruct_record(record_type, direct);

                // A fallback since we cannot really easily query whether a type
                // is primitive yet.
                return { convert_primitive_type(target_type, direct) };
            }
        };

        template< typename S, typename P >
        deconstructs_types( S &, P & ) -> deconstructs_types< S, P >;

        template< typename state_t, typename pattern_t >
        struct reconstructs_types
        {
            state_t &state;
            pattern_t &pattern;

            reconstructs_types(state_t &state, pattern_t &pattern)
                : state(state), pattern(pattern)
            {}

            /* `abi::DirectOp` related functions. */

            mlir::Value reconstruct_record(hl::RecordType record_type, abi::DirectOp direct)
            {
                return conv::abi::reconstruct_aggregate(pattern, direct,
                                                        record_type, state.rewriter);
            }

            mlir::Value convert_primitive_type(mlir_type target_type, abi::DirectOp direct)
            {
                VAST_CHECK(direct.getNumOperands() >= 1,
                           "abi.direct op should have > 1 operands: {0}", direct);

                // We need to reconstruct the type and we ?know? that arguments
                // simply need to be concated?
                VAST_CHECK(can_concat_as(pattern.dl, direct.getOperands().getTypes(),
                                         target_type),
                           "Cannot do concat when converting {0}", direct);

                return state.rewriter.template create< ll::Concat >(
                        direct.getLoc(),
                        target_type, direct.getOperands());
            }

            mlir::Value convert(mlir::Type res_type, abi::DirectOp direct)
            {
                auto target_type = hl::strip_value_category(res_type);
                if (target_type == direct.getOperand(0).getType())
                    return direct.getOperand(0);


                auto naked = hl::strip_elaborated(target_type);
                if (auto record_type = mlir::dyn_cast< hl::RecordType >(naked))
                    return reconstruct_record(record_type, direct);

                // A fallback since we cannot really easily query whether a type
                // is primitive yet.
                return convert_primitive_type(target_type, direct);
            }

            values handle(abi::DirectOp direct) &
            {
                // TODO(conv:abi): Can direct have more return types?
                VAST_ASSERT(direct.getNumResults() == 1 && direct.getNumOperands() > 0);
                auto var = state.rewriter.template create< ll::UninitializedVar >(
                        direct.getLoc(), direct.getResult().getType());

                std::vector< mlir::Value > init_values;
                for (auto res_type : direct.getResult().getTypes())
                    init_values.push_back(convert(res_type, direct));

                co_yield state.rewriter.template create< ll::InitializeVar >(
                        direct.getLoc(),
                        var.getResult().getType(),
                        var, init_values);
            }
        };

        template< typename S, typename P >
        reconstructs_types( S &, P & ) -> reconstructs_types< S, P >;

        struct epilogue : function_border_base< abi::EpilogueOp >
        {
            using base = function_border_base< op_t >;
            using op_t = typename base::op_t;

            using base::base;

            using state_capture = typename base::state_capture;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter) const override
            {
                return base::rewrite({ op, ops, rewriter });
            }

            // TODO(conv:abi): Copy & paste of prologue
            values match_on(abi::DirectOp direct, state_capture &state) const override
            {
                auto dtor = deconstructs_types(state, *this);
                for (auto v : dtor.handle(direct))
                    co_yield v;
            }
        };


        struct prologue : function_border_base< abi::PrologueOp >
        {
            using base = function_border_base< op_t >;
            using op_t = typename base::op_t;

            using base::base;

            using state_capture = typename base::state_capture;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter) const override
            {
                return base::rewrite({ op, ops, rewriter });
            }

            // TODO(conv:abi): Copy & paste of prologue
            values match_on(abi::DirectOp direct, state_capture &state) const override
            {
                auto ctor = reconstructs_types(state, *this);
                for (auto v : ctor.handle(direct))
                    co_yield v;
            }
        };

        struct call_args : call_border< abi::CallArgsOp >
        {
            using base = function_border_base< op_t >;
            using op_t = typename base::op_t;

            using base::base;

            using state_capture = typename base::state_capture;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter) const override
            {
                return base::rewrite({ op, ops, rewriter });
            }

            // TODO(conv:abi): Copy & paste of prologue
            values match_on(abi::DirectOp direct, state_capture &state) const override
            {
                auto dtor = deconstructs_types(state, *this);
                for (auto v : dtor.handle(direct))
                    co_yield v;
            }
        };

        struct call_rets : function_border_base< abi::CallRetsOp >
        {
            using base = function_border_base< op_t >;
            using op_t = typename base::op_t;

            using base::base;

            using state_capture = typename base::state_capture;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter) const override
            {
                return base::rewrite({ op, ops, rewriter });
            }

            // TODO(conv:abi): Copy & paste of prologue
            values match_on(abi::DirectOp direct, state_capture &state) const override
            {
                auto ctor = reconstructs_types(state, *this);
                for (auto v : ctor.handle(direct))
                    co_yield v;
            }
        };

        using wrappers = util::type_list< prologue, epilogue >;

    } // namespace
    } // namespace pattern

    struct LowerABI : ModuleConversionPassMixin< LowerABI, LowerABIBase >
    {

        using base = ModuleConversionPassMixin< LowerABI, LowerABIBase >;
        using config_t = typename base::config_t;

        static conversion_target create_conversion_target(mcontext_t &context)
        {
            conversion_target target(context);

            target.markUnknownOpDynamicallyLegal([](auto) { return true; } );
            return target;
        }

        void add_patterns(auto &config, const auto &dl)
        {
            config.patterns.template add< pattern::prologue >(dl, config.getContext());
            config.patterns.template add< pattern::epilogue >(dl, config.getContext());

            config.patterns.template add< pattern::call_args >(dl, config.getContext());
            config.patterns.template add< pattern::call_rets >(dl, config.getContext());

            config.target.template addIllegalOp< abi::PrologueOp >();
            config.target.template addIllegalOp< abi::EpilogueOp >();

            config.target.template addIllegalOp< abi::CallArgsOp >();
            config.target.template addIllegalOp< abi::CallRetsOp >();
        }

        // TODO(conv:abi): Neeeded hack for this to compile.
        template< typename pattern >
        static void add_pattern(config_t &config) {}

        // There is no helper we can use.
        void runOnOperation() override
        {
            auto &ctx   = getContext();
            auto config = config_t { rewrite_pattern_set(&ctx),
                                     create_conversion_target(ctx) };
            auto op     = this->getOperation();

            const auto &dl_analysis = this->template getAnalysis< mlir::DataLayoutAnalysis >();
            auto dl = dl_analysis.getAtOrAbove(op);

            add_patterns(config, dl);

            if (mlir::failed(base::apply_conversions(std::move(config))))
                return signalPassFailure();

            this->after_operation();
        }


        void populate_conversions(config_t &config)
        {
            base::populate_conversions_base< pattern::wrappers >(config);
        }
    };

} // namespace vast

std::unique_ptr< mlir::Pass > vast::createLowerABIPass()
{
    return std::make_unique< vast::LowerABI >();
}
