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

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/ABI/ABI.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/TypeSwitch.hpp"

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
        // TODO(conv:abi): Pull the utility walker into some shared header.
        template< typename T >
        gap::generator< T > walk(vast_module mod)
        {
            auto body = mod.getBody();
            if (!body)
                co_return;

            for (auto &op : *body)
                if (auto casted = mlir::dyn_cast< T >(op))
                    co_yield casted;
        }

        [[ maybe_unused ]] gap::generator< mlir::Type > field_types(hl::StructDeclOp op)
        {
            if (op.getFields().empty())
                co_return;

            for (auto &maybe_field : op.getOps())
            {
                // TODO(conv): So normally only `hl.field` should be present here,
                //             but currently also re-declarations of nested structures
                //             are here - add hard fail if the conversion fails in the future.
                if (mlir::isa< hl::StructDeclOp >(maybe_field))
                    continue;

                auto field_decl = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
                VAST_ASSERT(field_decl);
                co_yield field_decl.getType();
            }
        }

        [[ maybe_unused ]] auto field_types(mlir::Type t, vast_module mod)
        {
            auto type_name = hl::name_of_record(t);
            VAST_ASSERT(type_name);
            for (auto op : walk< hl::StructDeclOp >(mod))
                if (op.getName() == *type_name)
                    return field_types(op);
            VAST_UNREACHABLE("Was not able to fetch definition of type: {0}", t);
        }

        // [ `current_arg`, offset into `current_arg`, size of `current_arg` ]
        using arg_list_position = std::tuple< std::size_t, std::size_t, std::size_t >;

        template< typename op_t >
        struct abi_pattern_base : operation_conversion_pattern< op_t >
        {
            using base = operation_conversion_pattern< op_t >;
            using base::base;

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

            mlir::Value reconstruct_record(mlir::Type res_type, hl::RecordType record_type,
                                           abi::DirectOp direct, state_capture &state) const
            {
                return {};
            }

            mlir::Value convert(mlir::Type res_type, abi::DirectOp direct,
                                state_capture &state) const
            {
                auto lvalue = mlir::dyn_cast< hl::LValueType >(res_type);
                VAST_CHECK(lvalue, "Result type of abi.direct is no an lvalue. {0}", res_type);

                auto alloca = state.rewriter.create< ll::UninitializedVar >(direct.getLoc(),
                                                                            res_type);

                auto type = lvalue.getElementType();
                if (type == direct.getOperand(0).getType())
                {
                    return state.rewriter.create< ll::InitializeVar >(direct.getLoc(),
                                                                      res_type,
                                                                      alloca,
                                                                      direct.getOperand(0));
                }

                auto naked = hl::strip_elaborated(type);
                if (auto record_type = mlir::dyn_cast< hl::RecordType >(naked))
                    return reconstruct_record(res_type, record_type, direct, state);

                return {};
            }

            values match_on(abi::DirectOp direct, state_capture &state) const override
            {
                // TODO(conv:abi): Can direct have more return types?
                VAST_ASSERT(direct.getNumResults() == 1 && direct.getNumOperands() > 0);
                for (auto res_type : direct.getResult().getTypes())
                {
                    co_yield convert(res_type, direct, state);
                }
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
