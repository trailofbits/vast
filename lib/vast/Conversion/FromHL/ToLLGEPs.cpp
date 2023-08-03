// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Symbols.hpp"
#include "vast/Util/DialectConversion.hpp"

namespace vast
{
    namespace pattern
    {
        template< typename T >
        struct DoConversion {};

        template<>
        struct DoConversion< hl::RecordMemberOp > : util::State< hl::RecordMemberOp >
        {
            using util::State< hl::RecordMemberOp >::State;

            mlir::LogicalResult convert()
            {
                auto parent_type = operands.getRecord().getType();

                auto module_op = op->getParentOfType< vast_module >();
                if (!module_op)
                    return mlir::failure();

                auto struct_decl = hl::definition_of(parent_type, module_op);
                if (!struct_decl)
                    return mlir::failure();

                auto idx = hl::field_idx(op.getName(), *struct_decl);
                if (!idx)
                    return mlir::failure();

                auto gep = rewriter.create< ll::StructGEPOp >(
                        op.getLoc(),
                        op.getType(),
                        operands.getRecord(),
                        rewriter.getI32IntegerAttr(*idx),
                        op.getNameAttr());
                rewriter.replaceOp( op, { gep } );

                return mlir::success();
            }

        };

        using record_member_op = util::BasePattern< hl::RecordMemberOp, DoConversion >;

    } // namespace pattern

    struct HLToLLGEPsPass : HLToLLGEPsBase< HLToLLGEPsPass >
    {
        void runOnOperation() override
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            trg.markUnknownOpDynamicallyLegal( [](auto) { return true; } );
            trg.addIllegalOp< hl::RecordMemberOp >();

            mlir::RewritePatternSet patterns(&mctx);

            patterns.add< pattern::record_member_op >(&mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
                return signalPassFailure();
        }
    };
} // namespace vast


std::unique_ptr< mlir::Pass > vast::createHLToLLGEPsPass()
{
    return std::make_unique< vast::HLToLLGEPsPass >();
}
