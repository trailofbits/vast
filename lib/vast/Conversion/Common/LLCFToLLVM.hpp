// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Conversion/Common/Block.hpp"
#include "vast/Conversion/Common/Rewriter.hpp"

#include "Common.hpp"

namespace vast::conv::irstollvm::ll_cf
{
    struct br : base_pattern< ll::Br >
    {
        using base = base_pattern< ll::Br >;
        using base::base;

        using op_t = ll::Br;
        using adaptor_t = typename op_t::Adaptor;

        mlir::LogicalResult matchAndRewrite(
                    op_t op, adaptor_t ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            rewriter.create< LLVM::BrOp >(op.getLoc(), ops.operands(), op.dest());
            rewriter.eraseOp(op);

            return mlir::success();
        }

    };

    struct cond_br : base_pattern< ll::CondBr >
    {
        using base = base_pattern< ll::CondBr >;
        using base::base;

        using op_t = ll::CondBr;
        using adaptor_t = typename op_t::Adaptor;

        mlir::LogicalResult matchAndRewrite(
                    op_t op, adaptor_t ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            rewriter.create< LLVM::CondBrOp >(
                    op.getLoc(),
                    ops.cond(),
                    op.true_dest() , ops.true_operands(),
                    op.false_dest(), ops.false_operands());
            rewriter.eraseOp( op );

            return mlir::success();
        }

    };

    struct scope : base_pattern< ll::Scope >
    {
        using base = base_pattern< ll::Scope >;
        using base::base;

        using op_t = ll::Scope;
        using adaptor_t = typename op_t::Adaptor;

        mlir::LogicalResult replace_terminator(auto &rewriter, mlir::Block &block,
                                               mlir::Block &start, mlir::Block &end) const
        {
            auto &last = block.back();
            std::vector< mlir::Value > no_vals;

            if (auto ret = mlir::dyn_cast< ll::ScopeRet >(last)) {
                make_after_op< LLVM::BrOp >(rewriter, &last, last.getLoc(), no_vals, &end);
            } else if (auto ret = mlir::isa< ll::ScopeRecurse >(last)) {
                make_after_op< LLVM::BrOp >(rewriter, &last, last.getLoc(),
                                            no_vals, &start);
            } else if (auto ret = mlir::dyn_cast< ll::CondScopeRet >(last)) {
                make_after_op< LLVM::CondBrOp >(rewriter, &last, last.getLoc(),
                                                ret.cond(),
                                                ret.dest(), ret.operands(),
                                                &end, no_vals);
            } else {
                // Nothing to do (do not erase, since it is a standard branching).
                return mlir::success();
            }

            rewriter.eraseOp(&last);
            return mlir::success();
        }


        mlir::LogicalResult matchAndRewrite(
                    op_t op, adaptor_t ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto [head_block, tail_block] = split_at_op(op, rewriter);

            if (!op.start_block())
                return mlir::failure();

            for ( auto &block : op.body() )
            {
                auto s = replace_terminator(rewriter, block, *op.start_block(), *tail_block);
                if (mlir::failed(s))
                    return mlir::failure();
            }

            auto op_entry_block = &*op.body().begin();
            inline_region_blocks(rewriter, op.body(), mlir::Region::iterator(tail_block));

            rewriter.mergeBlocks(op_entry_block, head_block, std::nullopt);
            rewriter.eraseOp(op);
            return mlir::success();
        }
    };

    using conversions = util::type_list<
          cond_br
        , br
        , scope
    >;

} // namespace vast::conv::irstollvm::ll_cf
