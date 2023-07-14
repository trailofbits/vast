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

        template< typename op_t >
        struct abi_pattern_base : operation_conversion_pattern< op_t >
        {
            using base = operation_conversion_pattern< op_t >;

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

            auto bw(mlir_type type) const { return ::vast::bw(dl, type); }
            auto bw(mlir::Value val) const { return bw(val.getType()); }

        };

        // TODO(conv:abi): Parametrize by materializer, which will resolve what dialect
        //                is the target
        //                 * hl
        //                 * ll-vars
        //                 * llvm
        template< typename pattern, typename op_t >
        struct field_allocator
        {
            using self_t = field_allocator< pattern, op_t >;

            struct state_t
            {

                std::size_t arg_idx = 0;
                std::size_t offset = 0;

                const pattern &parent;
                op_t abi_op;

                state_t(const pattern &parent, op_t abi_op)
                    : parent(parent), abi_op(abi_op)
                {}

                state_t( const state_t &) = delete;
                state_t( state_t &&) = delete;

                state_t &operator=(state_t) = delete;

                auto arg() { return abi_op.getOperands()[arg_idx]; };
                auto bw(auto w) { return parent.bw(w); };

                bool fits(mlir_type type)
                {
                    return bw(arg()) >= offset + bw(type);
                };

                void advance()
                {
                    ++arg_idx;
                    offset = 0;
                }

                // TODO(conv:abi): Possibly add some debug prints to help us
                //                 debug some weird corner cases in the future?
                mlir::Value allocate(mlir_type type, auto &rewriter)
                {
                    auto start = offset;
                    offset += bw(type);
                    if (bw(type) == bw(arg()))
                        return arg();

                    return rewriter.template create< ll::Extract >(
                            abi_op.getLoc(), type, arg(),
                            start, offset);
                };
            };

            state_t &state;
            vast_module mod;
            std::vector< mlir::Value > partials;

            field_allocator(state_t &state, vast_module mod)
                : state(state),
                  mod(mod)
            {}

            static state_t mk_state(const pattern &parent, op_t abi_op)
            {
                return state_t(parent, abi_op);
            }

            bool needs_nesting(mlir_type type) const
            {
                return contains_subtype< hl::RecordType >(type);
            }

            mlir::Value run_on(mlir_type root_type, auto &rewriter)
            {
                auto handle_type = [&](mlir_type field_type) -> mlir::Value
                {
                    if (needs_nesting(field_type))
                        return self_t(state, mod).run_on(field_type, rewriter);

                    if (!state.fits(field_type))
                        state.advance();
                    return state.allocate(field_type, rewriter);
                };

                for (auto field_type : vast::hl::field_types(root_type, mod))
                    partials.push_back(handle_type(field_type));

                // Make the thing;
                return make_aggregate(root_type, partials, rewriter);
            }

            mlir::Value make_aggregate(mlir_type type, const auto &partials,
                                       auto &rewriter)
            {
                return rewriter.template create< hl::InitListExpr >(
                        state.abi_op.getLoc(), type, partials).getResult(0);
            }

        };

        struct prologue : abi_pattern_base< abi::PrologueOp >
        {
            using op_t = abi::PrologueOp;
            using base = abi_pattern_base< op_t >;

            using base::base;

            using state_capture = base::state_capture;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter) const override
            {
                std::vector< mlir::Value > to_replace;
                for (auto &nested : op.getBody().getOps())
                {
                    for (auto converted : base::dispatch(&nested, { op, ops, rewriter }))
                        to_replace.push_back(converted);
                }

                rewriter.replaceOp(op, to_replace);
                return mlir::success();
            }

            mlir::Value reconstruct_record(hl::RecordType record_type,
                                           abi::DirectOp direct, state_capture &state) const
            {
                auto mod = direct->getParentOfType< vast_module >();

                using allocator = field_allocator< prologue, abi::DirectOp >;
                auto allocator_state = allocator::mk_state(*this, direct);

                auto out = allocator(allocator_state, mod).run_on(record_type,
                                                                  state.rewriter);
                return out;
            }

            mlir::Value convert_primitve_type(mlir_type target_type, abi::DirectOp direct,
                                              state_capture &state) const
            {
                VAST_CHECK(direct.getNumOperands() >= 1,
                           "abi.direct op should have > 1 operands: {0}", direct);

                // We need to reconstruct the type and we ?know? that arguments
                // simply need to be concated?
                VAST_CHECK(can_concat_as(dl, direct.getOperands().getTypes(), target_type),
                           "Cannot do concat when converting {0}", direct);

                return state.rewriter.create< ll::Concat >(
                        direct.getLoc(),
                        target_type, direct.getOperands());
            }

            mlir::Value convert(mlir::Type res_type, abi::DirectOp direct,
                                state_capture &state) const
            {
                auto lvalue = mlir::dyn_cast< hl::LValueType >(res_type);
                VAST_CHECK(lvalue, "Result type of abi.direct is no an lvalue. {0}", res_type);


                auto target_type = lvalue.getElementType();
                if (target_type == direct.getOperand(0).getType())
                    return direct.getOperand(0);


                auto naked = hl::strip_elaborated(target_type);
                if (auto record_type = mlir::dyn_cast< hl::RecordType >(naked))
                    return reconstruct_record( record_type, direct, state);

                // A fallback since we cannot really easily query whether a type
                // is primitive yet.
                return convert_primitve_type(target_type, direct, state);
            }

            values match_on(abi::DirectOp direct, state_capture &state) const override
            {
                // TODO(conv:abi): Can direct have more return types?
                VAST_ASSERT(direct.getNumResults() == 1 && direct.getNumOperands() > 0);
                auto var = state.rewriter.create< ll::UninitializedVar >(
                        direct.getLoc(), direct.getResult().getType());

                std::vector< mlir::Value > init_values;
                for (auto res_type : direct.getResult().getTypes())
                    init_values.push_back(convert(res_type, direct, state));

                co_yield state.rewriter.create< ll::InitializeVar >(direct.getLoc(),
                                                                    var.getResult().getType(),
                                                                    var, init_values);
            }
        };

        using wrappers = util::type_list< prologue >;

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

            config.target.template addIllegalOp< abi::PrologueOp >();
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
