// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{
    namespace
    {
        [[maybe_unused]] std::size_t size(mlir::Region &region)
        {
            return std::distance(region.begin(), region.end());
        }
        [[maybe_unused]] std::size_t size(mlir::Block &block)
        {
            return std::distance(block.begin(), block.end());
        }
    }



    void copy_block(auto &b, auto &rewriter, auto &region, auto loc)
    {
        auto ip = rewriter.getInsertionBlock();
        auto dst_region = ip->getParent();

        rewriter.inlineRegionBefore(region, *dst_region, dst_region->begin());
        VAST_ASSERT(size(*dst_region) <= 2);
        if (size(*dst_region) == 2)
            rewriter.mergeBlocks(&*dst_region->begin(), &*std::next(dst_region->begin()),
                                 llvm::None);

        if (size(*dst_region->begin()) == 0)
        {
            rewriter.setInsertionPointToStart(&*dst_region->begin());
            rewriter.template create< mlir::scf::YieldOp >(loc);
        } else {
            rewriter.setInsertionPointToEnd(&*dst_region->begin());
            rewriter.template create< mlir::scf::YieldOp >(loc);
        }

    }
    template< typename T >
    auto inline_cond_region(auto src, auto &rewriter)
    {
        auto up = src->getParentRegion();
        VAST_ASSERT(up != nullptr);

        auto &copy = src.condRegion();
        VAST_ASSERT(size(copy) == 1);

        rewriter.cloneRegionBefore(copy, *up, up->end());
        rewriter.mergeBlockBefore(&up->back(), src);

        return mlir::dyn_cast< T >(*std::prev(mlir::Block::iterator(src)));
    }

    struct LowerIfOp : mlir::ConvertOpToLLVMPattern< hl::IfOp >
    {
        using Base = mlir::ConvertOpToLLVMPattern< hl::IfOp >;
        using Base::Base;

        void make_if_block(mlir::Region &old_region, mlir::Region &new_region,
                           mlir::Location loc, auto &rewriter) const
        {
            rewriter.inlineRegionBefore(old_region, new_region, new_region.begin());

            rewriter.eraseBlock(&new_region.back());

            VAST_ASSERT(size(new_region) == 1);
            auto &block = new_region.front();
            VAST_ASSERT(block.getNumArguments() == 0);


            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(&block);
            std::vector< mlir::Value > vals;
            rewriter.template create< mlir::scf::YieldOp >(loc, vals);
        }

        mlir::LogicalResult matchAndRewrite(
                hl::IfOp op, hl::IfOp::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto yield = inline_cond_region< hl::CondYieldOp >(op, rewriter);
            rewriter.setInsertionPointAfter(yield);

            mlir::scf::IfOp scf_if_op = rewriter.create< mlir::scf::IfOp >(
                    op.getLoc(), std::vector< mlir::Type >{}, yield.getOperand(), true);
            make_if_block(op.thenRegion(), scf_if_op.getThenRegion(), op.getLoc(), rewriter);
            make_if_block(op.elseRegion(), scf_if_op.getElseRegion(), op.getLoc(), rewriter);

            rewriter.eraseOp(yield);
            rewriter.eraseOp(op);

            return mlir::success();
        }
    };

    namespace pattern
    {

        struct l_while : mlir::ConvertOpToLLVMPattern< hl::WhileOp >
        {
            using Base = mlir::ConvertOpToLLVMPattern< hl::WhileOp >;
            using Base::Base;


            mlir::Block &do_inline(mlir::Region &src, mlir::Region &dst,
                                   mlir::ConversionPatternRewriter &rewriter) const
            {
                rewriter.createBlock(&dst);
                rewriter.cloneRegionBefore(src, &dst.back());
                rewriter.eraseBlock(&dst.back());

                return dst.back();
            }

            template< typename T >
            std::optional< T > fetch_terminator(mlir::Block &block) const
            {
                auto out = mlir::dyn_cast< T >(block.getTerminator());
                if (!out)
                    return {};
                return out;
            }

            mlir::LogicalResult before_region(mlir::Block &dst,
                                              mlir::ConversionPatternRewriter &rewriter) const
            {
                auto cond_yield = fetch_terminator< hl::CondYieldOp >(dst);
                if (!cond_yield)
                    return mlir::failure();

                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointAfter(*cond_yield);
                rewriter.create< mlir::scf::ConditionOp >(
                        cond_yield->getLoc(),
                        cond_yield->result(),
                        dst.getParent()->front().getArguments());

                rewriter.eraseOp(*cond_yield);
                return mlir::success();
            }

            mlir::LogicalResult after_region(mlir::Block &block, mlir::Block &before,
                                             mlir::Location loc,
                                             mlir::ConversionPatternRewriter &rewriter) const
            {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToEnd(&block);
                rewriter.create< mlir::scf::YieldOp >(loc, before.getArguments());

                return mlir::success();
            }

            mlir::LogicalResult matchAndRewrite(
                    hl::WhileOp op, hl::WhileOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto scf_while_op = rewriter.create< mlir::scf::WhileOp >(
                        op.getLoc(),
                        std::vector< mlir::Type >{},
                        std::vector< mlir::Value >{});
                auto &before = do_inline(op.condRegion(), scf_while_op.getBefore(), rewriter);
                auto &after = do_inline(op.bodyRegion(), scf_while_op.getAfter(), rewriter);

                if (mlir::failed(before_region(before, rewriter)) ||
                    mlir::failed(after_region(after, before, op.getLoc(), rewriter)))
                {
                    return mlir::failure();
                }

                rewriter.eraseOp(op);
                return mlir::success();
            }
        };

    } // namespace pattern


    struct LowerHighLevelControlFlowPass : LowerHighLevelControlFlowBase< LowerHighLevelControlFlowPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelControlFlowPass::runOnOperation()
    {
        auto op = this->getOperation();
        auto &mctx = this->getContext();

        mlir::ConversionTarget trg(mctx);
        trg.addIllegalOp< hl::IfOp >();
        trg.addIllegalOp< hl::WhileOp >();
        trg.addIllegalOp< hl::CondYieldOp >();
        trg.addLegalOp< mlir::scf::IfOp >();
        trg.addLegalOp< mlir::scf::YieldOp >();
        trg.addLegalOp< mlir::scf::ConditionOp >();
        trg.markUnknownOpDynamicallyLegal([](auto) { return true; });

        mlir::RewritePatternSet patterns(&mctx);

        mlir::LowerToLLVMOptions llvm_opts{ &mctx };
        const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();

        auto tc = mlir::LLVMTypeConverter(&mctx, llvm_opts, &dl_analysis);
        patterns.add< LowerIfOp >(tc);
        patterns.add< pattern::l_while >(tc);
        if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
            return signalPassFailure();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelControlFlowPass()
{
  return std::make_unique< LowerHighLevelControlFlowPass >();
}
