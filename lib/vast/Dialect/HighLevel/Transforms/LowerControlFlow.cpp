// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

namespace vast::hl
{
    template< typename T >
    T inline_cond_region(auto src, auto &rewriter)
    {
        auto &cond_region = src.condRegion();
        auto &init_block = cond_region.back();

        auto terminator = mlir::dyn_cast< T >(init_block.getTerminator());
        VAST_ASSERT(terminator);

        rewriter.inlineRegionBefore(cond_region, src->getBlock());
        auto ip = std::next(mlir::Block::iterator(src));
        VAST_ASSERT(ip != src->getBlock()->end());

        rewriter.mergeBlockBefore(&init_block, &*ip);
        return terminator;
    }

    namespace
    {
        std::size_t size(mlir::Region &region)
        {
            return std::distance(region.begin(), region.end());
        }
        std::size_t size(mlir::Block &block)
        {
            return std::distance(block.begin(), block.end());
        }
    }



    void copy_block(auto &b, auto &rewriter, auto &region, auto loc)
    {
        auto ip = rewriter.getInsertionBlock();
        auto dst_region = ip->getParent();

        rewriter.inlineRegionBefore(region, *dst_region, dst_region->begin());
        llvm::errs() << size(*dst_region) << " AAA\n"; llvm::errs().flush();
        VAST_ASSERT(size(*dst_region) <= 2);
        if (size(*dst_region) == 2)
            rewriter.mergeBlocks(&*dst_region->begin(), &*std::next(dst_region->begin()),
                                 llvm::None);

        //auto terminator = dst_region->begin()->getTerminator();
        //rewriter.setInsertionPoint(terminator);
        //if (auto yield = terminator.dyn_cast< hl::ValueYieldOp >/
        //rewriter.mergeBlockBefore(&init_block, std::next(mlir::Block::iterator(region)));
        if (size(*dst_region->begin()) == 0)
        {
            rewriter.setInsertionPointToStart(&*dst_region->begin());
            rewriter.template create< mlir::scf::YieldOp >(loc);
        } else {
            rewriter.setInsertionPointToEnd(&*dst_region->begin());
            rewriter.template create< mlir::scf::YieldOp >(loc);
        }

    }

    struct LowerIfOp : mlir::ConvertOpToLLVMPattern< hl::IfOp >
    {
        using Base = mlir::ConvertOpToLLVMPattern< hl::IfOp >;
        using Base::Base;

        mlir::LogicalResult matchAndRewrite(
                hl::IfOp op, hl::IfOp::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto yield = inline_cond_region< hl::CondYieldOp >(op, rewriter);
            rewriter.setInsertionPoint(yield);
            auto c = yield.getOperand();

            mlir::scf::IfOp if_op = rewriter.create< mlir::scf::IfOp >(
                    op.getLoc(), yield.getOperand(),
                    [&](auto &b, auto) { copy_block(b, rewriter, op.thenRegion(), op.getLoc()); },
                    [&](auto &b, auto) { copy_block(b, rewriter, op.elseRegion(), op.getLoc()); }
                    );
            rewriter.eraseOp(yield);
            rewriter.eraseOp(op);

            return mlir::success();
        }
    };


    struct LowerHighLevelControlFlowPass : LowerHighLevelControlFlowBase< LowerHighLevelControlFlowPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelControlFlowPass::runOnOperation() {}
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelControlFlowPass()
{
  return std::make_unique< LowerHighLevelControlFlowPass >();
}
