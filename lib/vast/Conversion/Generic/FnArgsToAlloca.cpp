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

namespace vast::conv
{
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
        }


        void arg_to_alloca(auto arg, auto &block, auto &bld) const
        {
            if (!needs_lowering(arg))
                return;

            // Now this is a question. Should this be lvalue or pointer?
            // It depends where in the pipeline we are? lvalue is probably safer
            // default for now.
            auto lvalue_type = hl::LValueType::get(bld.getContext(), arg.getType());
            auto lowered = bld.template create< ll::ArgAlloca >(
                arg.getLoc(), lvalue_type, arg);

            arg.replaceAllUsesWith(lowered);
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
