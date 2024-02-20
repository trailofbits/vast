// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Common.hpp"

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"
#include "vast/Conversion/TypeConverters/HLToStd.hpp"

namespace vast::conv
{
    namespace
    {
        struct strip_lvalue : tc::base_type_converter,
                              tc::mixins< strip_lvalue >,
                              tc::CoreToStd< strip_lvalue >
        {
            mcontext_t &mctx;

            strip_lvalue(mcontext_t &mctx) : mctx(mctx)
            {
                init();
            }

            auto convert_lvalue()
            {
                return [&](hl::LValueType type)
                {
                    return Maybe(type.getElementType())
                        .and_then(convert_type_to_type())
                        .unwrap()
                        .template take_wrapped< maybe_type_t >();
                };
            }

            void init()
            {
                addConversion([&](mlir_type t) { return t; });
                tc::CoreToStd< strip_lvalue >::init();
                addConversion(convert_lvalue());
            }

        };

        using strip_lvalue_pattern = tc::generic_type_converting_pattern< strip_lvalue >;

    } // namespace

    struct FnArgsToAllocaPass : FnArgsToAllocaBase< FnArgsToAllocaPass >
    {
        using base = FnArgsToAllocaBase< FnArgsToAllocaPass >;

        void runOnOperation() override
        {
            auto root = getOperation();

            // TODO(conv): It would be much better if this pass could run on
            //             `mlir::FunctionOpInterface` instead of module.
            auto lower = [&](mlir::FunctionOpInterface fn)
            {
                lower_args_to_alloca(fn);
            };

            root->walk(lower);

            // Now proceed to update function types by stripping lvalues
            auto &mctx = getContext();

            auto tc = strip_lvalue(mctx);
            auto trg = mlir::ConversionTarget(mctx);

            auto is_fn = [&](operation op)
            {
                auto as_fn = mlir::dyn_cast< mlir::FunctionOpInterface >(op);
                if (!as_fn)
                    return true;

                for (auto arg_type : as_fn.getArgumentTypes())
                    if (!tc.isLegal(arg_type))
                        return false;
                return true;
            };
            trg.markUnknownOpDynamicallyLegal(is_fn);

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< strip_lvalue_pattern >(tc, mctx);

            if (mlir::failed(mlir::applyPartialConversion(root, trg, std::move(patterns))))
                return signalPassFailure();
        }


        void arg_to_alloca(auto arg, auto &block, auto &bld) const
        {
            if (!needs_lowering(arg))
                return;

            auto lowered = bld.template create< ll::ArgAlloca >(
                arg.getLoc(), arg.getType(), arg);

            arg.replaceAllUsesWith(lowered);
            lowered->setOperand(0, arg);
        }

        bool needs_lowering(auto arg) const
        {
            for (auto user : arg.getUsers())
                if (!mlir::isa< ll::ArgAlloca >(user))
                    return true;
            return false;
        }

        void lower_args_to_alloca(mlir::FunctionOpInterface fn)
        {
            if (!fn || fn.empty())
                return;

            auto &block = fn.front();
            if (!block.isEntryBlock())
                return;

            // We don't care about guards
            mlir::OpBuilder bld(&getContext());
            bld.setInsertionPointToStart(&block);
            for (auto arg : block.getArguments())
                arg_to_alloca(arg, block, bld);
        }

    };
} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createFnArgsToAllocaPass()
{
    return std::make_unique< vast::conv::FnArgsToAllocaPass >();
}
