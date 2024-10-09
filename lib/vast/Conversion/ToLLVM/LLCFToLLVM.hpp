// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Symbols.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Conversion/Common/Block.hpp"
#include "vast/Conversion/Common/Rewriter.hpp"

namespace vast::conv::irstollvm::cf
{
    namespace LLVM = mlir::LLVM;

    struct br : operation_conversion_pattern< ll::Br >
    {
        using base = operation_conversion_pattern< ll::Br >;
        using base::base;

        using adaptor_t = typename ll::Br::Adaptor;

        logical_result matchAndRewrite(
            ll::Br op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            rewriter.create< LLVM::BrOp >(op.getLoc(), ops.getOperands(), op.getDest());
            rewriter.eraseOp(op);
            return mlir::success();
        }
    };

    struct cond_br : operation_conversion_pattern< ll::CondBr >
    {
        using base = operation_conversion_pattern< ll::CondBr >;
        using base::base;

        using op_t = ll::CondBr;
        using adaptor_t = typename op_t::Adaptor;

        logical_result matchAndRewrite(
            ll::CondBr op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            rewriter.create< LLVM::CondBrOp >(
                op.getLoc(),
                ops.getCond(),
                op.getTrueDest() , ops.getTrueOperands(),
                op.getFalseDest(), ops.getFalseOperands()
            );
            rewriter.eraseOp( op );
            return mlir::success();
        }

    };

    template< typename op_t >
    struct region_to_block_conversion_pattern
        : operation_conversion_pattern< op_t >
    {
        using base = operation_conversion_pattern< op_t >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        // TODO(conv:irstollvm): Separate terminator types should be CRTP hookable.
        logical_result replace_terminator(
            auto &rewriter, block_t &block, block_t &start, block_t &end
        ) const {
            auto &last = block.back();
            std::vector< mlir::Value > no_vals;

            if (mlir::isa< ll::ScopeRet >(last)) {
                make_after_op< LLVM::BrOp >(rewriter, &last, last.getLoc(), no_vals, &end);
            } else if (mlir::isa< ll::ScopeRecurse >(last)) {
                make_after_op< LLVM::BrOp >(
                    rewriter, &last, last.getLoc(),no_vals, &start
                );
            } else if (auto ret = mlir::dyn_cast< ll::CondScopeRet >(last)) {
                make_after_op< LLVM::CondBrOp >(
                    rewriter, &last, last.getLoc(),
                    ret.getCond(),
                    ret.getDest(), ret.getDestOperands(),
                    &end, no_vals
                );
            } else {
                // Nothing to do (do not erase, since it is a standard branching).
                return mlir::success();
            }

            rewriter.eraseOp(&last);
            return mlir::success();
        }


        logical_result handle_multiblock(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const {
            auto [head_block, tail_block] = split_at_op(op, rewriter);

            if (!start_block(op))
                return mlir::failure();

            for ( auto &block : op.getBody() )
            {
                auto s = replace_terminator(rewriter, block, *start_block(op), *tail_block);
                if (mlir::failed(s))
                    return mlir::failure();
            }

            auto op_entry_block = &*op.getBody().begin();
            rewriter.setInsertionPointToEnd(head_block);
            rewriter.create< mlir::LLVM::BrOp >(
                op.getLoc(), std::vector< mlir::Value >{}, op_entry_block
            );

            inline_region_blocks(rewriter, op.getBody(), mlir::Region::iterator(tail_block));

            rewriter.eraseOp(op);
            return mlir::success();
        }

        logical_result handle_singleblock(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const {
            auto parent = op->getParentRegion();

            rewriter.inlineRegionBefore(op.getBody(), *parent, parent->end());

            // splice newly created translation unit block in the module
            auto &unit_block = parent->back();
            rewriter.inlineBlockBefore(&unit_block, op, {});

            rewriter.eraseOp(op);
            return mlir::success();
        }

        // TODO(conv:irstollvm): Should be handled on the operation api level.
        block_t *start_block(op_t op) const { return &op.getBody().front(); }

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            if (op.getBody().empty()) {
                rewriter.eraseOp(op);
                return logical_result::success();
            }

            // If we do not have any branching inside, we can just "inline" the op.
            if (op.getBody().hasOneBlock()) {
                return handle_singleblock(op, ops, rewriter);
            }

            return handle_multiblock(op, ops, rewriter);
        }
    };

    using patterns = util::type_list< cond_br, br >;

} // namespace vast::conv::irstollvm::ll_cf
