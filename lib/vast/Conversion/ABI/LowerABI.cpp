// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include <mlir/Rewrite/PatternApplicator.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/ABI/Common.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"

#include "vast/Dialect/Core/Func.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/ABI/ABI.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/TypeSwitch.hpp"
#include "vast/Util/TypeUtils.hpp"

#include "vast/Dialect/ABI/ABIOps.hpp"

#include "vast/Conversion/Common/Mixins.hpp"

#include <iostream>
#include <unordered_map>

#include <gap/coro/generator.hpp>

#include "vast/Conversion/ABI/AggregateTypes.hpp"
#include "vast/Conversion/Common/Block.hpp"

namespace vast
{
    using values        = gap::generator< mlir::Value >;
    using frozen_values = std::vector< mlir::Value >;

    namespace pattern
    {
    namespace
    {
        auto query_bw(const auto &dl, mlir_type type)
        {
            return dl.getTypeSizeInBits(hl::strip_value_category(type));
        }

        auto query_bw(const auto &dl, auto type_range)
        {
            std::size_t acc = 0;
            for (auto type : type_range)
                acc += query_bw(dl, type);
            return acc;
        }

        bool can_concat_as(const auto &dl, auto type_range, auto target)
        {
            return query_bw(dl, target) == query_bw(dl, type_range);
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
            virtual values match_on(abi::DirectOp direct, state_capture state) const = 0;
            virtual values match_on(abi::IndirectOp indirect, state_capture state) const = 0;

            virtual values match_on(mlir::Operation *op, state_capture state) const
            {
                co_return;
            }

            values dispatch(mlir::Operation *op, state_capture state) const
            {
                if (auto direct = mlir::dyn_cast< abi::DirectOp >(op))
                    return this->match_on(direct, state);
                if (auto indirect = mlir::dyn_cast< abi::IndirectOp >(op))
                    return this->match_on(indirect, state);
                return this->match_on(op, state);
            }

            static auto get(auto &&thing, long idx)
            {
                VAST_ASSERT(std::distance(thing.begin(), thing.end()) > idx);
                return *std::next(thing.begin(), idx);
            }

            auto bw(mlir_type type) const
            {
                return query_bw(dl, hl::strip_value_category(type));
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

                VAST_PATTERN_CHECK(state.op.getNumResults() == to_replace.size(),
                                   "Incorrect replacement: {0} != {1} of op {2}.",
                                   state.op.getNumResults(), to_replace.size(), state.op);
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
                    co_yield init_values[i];
            }


            auto deconstruct_record(hl::RecordType record_type, abi::DirectOp direct)
            {
                // So we are going to emit a bunch `hl.member` which are
                // semantically geps. These need an operand, that is lvalue.
                // First, we try to see if lvalue can be fetched one step backwards
                // in DF.
                // IF not we assert for now - possible behaviour is to introduce custom
                // `ll` uninitialized variable (as they don't need names) and then
                // simply hope later on this will get obliterated during some form
                // of mem2reg.
                auto as_ptr = [&]() -> mlir::Operation *
                {
                    VAST_ASSERT(direct.getOperand(0).getDefiningOp()->getNumResults() == 1);
                    auto mctx = direct.getContext();
                    auto type = hl::PointerType::get(
                        mctx, direct.getOperand(0).getType());
                    auto memory = state.rewriter.template create< ll::Alloca >(
                        direct.getLoc(), type);
                    state.rewriter.template create< ll::Store >(
                        direct.getLoc(), direct.getOperand(0), memory);
                    return memory;
                }();

                return conv::abi::deconstruct_aggregate(pattern, direct, as_ptr,
                                                        state.rewriter);
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
                // Not invoking yet, since I may do something if value is lvalue.
                auto convert_values = [&]()
                {
                    std::vector< mlir::Value > init_values;
                    for (auto res_type : direct.getResult().getTypes())
                        init_values.push_back(convert(res_type, direct));
                    return init_values;
                };

                if (!mlir::isa< hl::LValueType >(direct.getResult().getTypes()[0]))
                {
                    for (auto v : convert_values())
                        co_yield v;
                    co_return;
                }

                auto var = state.rewriter.template create< ll::UninitializedVar >(
                        direct.getLoc(), direct.getResult().getType());

                co_yield state.rewriter.template create< ll::InitializeVar >(
                        direct.getLoc(),
                        var.getResult().getType(),
                        var, convert_values());
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
            values match_on(abi::DirectOp direct, state_capture state) const override
            {
                auto dtor = deconstructs_types(state, *this);
                for (auto v : dtor.handle(direct))
                    co_yield v;
            }

            values match_on(abi::IndirectOp indirect, state_capture state) const override
            {
                VAST_UNREACHABLE("Not implemented");
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
            // TODO(conv:abi): Should this use `gap::recursive_generator` instead?
            values match_on(abi::DirectOp direct, state_capture state) const override
            {
                auto ctor = reconstructs_types(state, *this);
                for (auto v : ctor.handle(direct))
                    co_yield v;
            }

            values match_on(abi::IndirectOp indirect, state_capture state) const override
            {
                co_yield state.rewriter.template create< ll::Load >(
                    indirect.getLoc(),
                    indirect.getResult().getType(),
                    indirect.getValue());

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
            values match_on(abi::DirectOp direct, state_capture state) const override
            {
                auto dtor = deconstructs_types(state, *this);
                for (auto v : dtor.handle(direct))
                    co_yield v;
            }

            values match_on(abi::IndirectOp indirect, state_capture state) const override
            {
                auto loc = indirect.getLoc();
                auto mctx = indirect.getContext();
                auto type = hl::PointerType::get(mctx, indirect.getValue().getType());

                auto var = state.rewriter.template create< ll::Alloca >(
                    indirect.getLoc(), type);
                // Now we initilizae before yielding the ptr
                state.rewriter.template create< ll::Store >(loc, indirect.getValue(), var);
                co_yield var;
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
            values match_on(abi::DirectOp direct, state_capture state) const override
            {
                auto ctor = reconstructs_types(state, *this);
                for (auto v : ctor.handle(direct))
                    co_yield v;
            }

            values match_on(abi::IndirectOp, state_capture ) const override
            {
                VAST_UNREACHABLE("Not implemented.");
            }
        };

        struct call_exec : operation_conversion_pattern< abi::CallExecutionOp >
        {
            using base = operation_conversion_pattern< abi::CallExecutionOp >;
            using op_t = abi::CallExecutionOp;

            using base::base;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter) const override
            {
                auto parent = op->getParentRegion();
                rewriter.inlineRegionBefore(op.getBody(), *parent, parent->end());
                auto &unit_block = parent->back();
                rewriter.inlineBlockBefore(&unit_block, op, {});

                auto previous_inst = &*std::prev(mlir::Block::iterator(op));
                auto yield = mlir::dyn_cast< abi::YieldOp >(previous_inst);

                VAST_ASSERT(yield);
                rewriter.replaceOp(op, yield.getOperands());
                rewriter.eraseOp(yield);

                return mlir::success();
            }
        };

        struct call : operation_conversion_pattern< abi::CallOp >
        {
            using base = operation_conversion_pattern< abi::CallOp >;
            using op_t = abi::CallOp;

            using base::base;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter) const override
            {
                auto new_op = rewriter.create< hl::CallOp >(
                        op.getLoc(),
                        op.getCallee(),
                        op.getResults().getTypes(),
                        ops.getOperands());
                rewriter.replaceOp(op, new_op.getResults());
                return mlir::success();
            }
        };

        struct function : operation_conversion_pattern< abi::FuncOp >
        {
            using base      = operation_conversion_pattern< abi::FuncOp >;
            using base::base;

            using op_t      = abi::FuncOp;
            using adaptor_t = abi::FuncOp::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter) const override
            {
                auto name = op.getName();
                if (!name.consume_front(conv::abi::abi_func_name_prefix))
                    return mlir::failure();

                return core::convert_and_replace_function< ll::FuncOp >(op, rewriter, name);
            }
        };

        using wrappers = util::type_list< prologue, epilogue >;

    } // namespace
    } // namespace pattern

    struct LowerABI : ConversionPassMixin< LowerABI, LowerABIBase >
    {
        using base = ConversionPassMixin< LowerABI, LowerABIBase >;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });
            return target;
        }

        void add_patterns(base_conversion_config &cfg, const auto &dl)
        {
            auto ctx = cfg.patterns.getContext();
            cfg.patterns.template add< pattern::prologue >(dl, ctx);
            cfg.patterns.template add< pattern::epilogue >(dl, ctx);

            cfg.patterns.template add< pattern::call_args >(dl, ctx);
            cfg.patterns.template add< pattern::call_rets >(dl, ctx);

            cfg.patterns.template add< pattern::call >(ctx);
            cfg.patterns.template add< pattern::call_exec >(ctx);

            cfg.patterns.template add< pattern::function >(ctx);

            cfg.target.template addIllegalOp< abi::PrologueOp >();
            cfg.target.template addIllegalOp< abi::EpilogueOp >();

            cfg.target.template addIllegalOp< abi::CallArgsOp >();
            cfg.target.template addIllegalOp< abi::CallRetsOp >();

            cfg.target.template addIllegalOp< abi::CallOp >();
            cfg.target.template addIllegalOp< abi::CallExecutionOp >();

            cfg.target.template addIllegalOp< abi::FuncOp >();
        }

        void runOnOperation() override
        {
            auto &ctx   = getContext();
            base_conversion_config cfg = {
                rewrite_pattern_set(&ctx),
                create_conversion_target(ctx)
            };

            auto op     = this->getOperation();

            const auto &dl_analysis = this->template getAnalysis< mlir::DataLayoutAnalysis >();
            auto dl = dl_analysis.getAtOrAbove(op);

            add_patterns(cfg, dl);

            if (mlir::failed(base::apply_conversions(std::move(cfg)))) {
                return signalPassFailure();
            }
        }
    };

} // namespace vast

std::unique_ptr< mlir::Pass > vast::createLowerABIPass()
{
    return std::make_unique< vast::LowerABI >();
}
