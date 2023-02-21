// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/DialectConversion.hpp"

#include "../PassesDetails.hpp"

namespace vast
{
    namespace
    {
        struct func : OpConversionPattern< hl::FuncOp >
        {
            using parent_t = OpConversionPattern< hl::FuncOp >;
            using parent_t::parent_t;

            mlir::LogicalResult matchAndRewrite(
                    hl::FuncOp op,
                    hl::FuncOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                mlir::SmallVector< mlir::DictionaryAttr, 8 > arg_attrs;
                mlir::SmallVector< mlir::NamedAttribute, 8 > other_attrs;
                op.getAllArgAttrs(arg_attrs);

                mlir::func::FuncOp lowered = rewriter.create< mlir::func::FuncOp >(
                        op.getLoc(),
                        op.getName(),
                        op.getFunctionType(),
                        other_attrs,
                        arg_attrs
                );
                lowered.setVisibility(op.getVisibility());
                rewriter.inlineRegionBefore(op.getBody(),
                                            lowered.getBody(),
                                            lowered.end());

                rewriter.eraseOp(op);

                return mlir::success();
            }
        };

        struct ret : OpConversionPattern< hl::ReturnOp >
        {
            using parent_t = OpConversionPattern< hl::ReturnOp >;
            using parent_t::parent_t;


            mlir::LogicalResult matchAndRewrite(
                    hl::ReturnOp op,
                    hl::ReturnOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.create< mlir::func::ReturnOp >(op.getLoc(),
                                                        ops.getResult());

                rewriter.eraseOp(op);
                return mlir::success();
            }
        };

    } // namespace

    struct HLFuncToFuncPass : HLFuncToFuncBase< HLFuncToFuncPass >
    {
        void runOnOperation() override
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            trg.addIllegalOp< hl::FuncOp >();
            trg.addIllegalOp< hl::ReturnOp >();
            trg.markUnknownOpDynamicallyLegal([](auto){ return true; });

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< func >(patterns.getContext());
            patterns.add< ret >(patterns.getContext());

            if (mlir::failed(mlir::applyPartialConversion(
                             op, trg, std::move(patterns))))
            {
                return signalPassFailure();
            }
        }

    };

    std::unique_ptr< mlir::Pass > createHLFuncToFuncPass()
    {
        return std::make_unique< HLFuncToFuncPass >();
    }
} // namespace vast
