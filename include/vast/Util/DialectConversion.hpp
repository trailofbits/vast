// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

namespace vast::util
{
    template< typename Op >
    struct State
    {
        using rewriter_t = mlir::ConversionPatternRewriter;

        Op op;
        typename Op::Adaptor operands;
        rewriter_t &rewriter;

        State(Op op, typename Op::Adaptor operands, rewriter_t &rewriter)
            : op(op), operands(operands), rewriter(rewriter)
        {}

        static mlir::Block &solo_block(mlir::Region &region)
        {
            VAST_ASSERT(region.hasOneBlock());
            return *region.begin();
        }
    };

    template< typename Op, template< typename > class Impl >
    struct BasePattern : mlir::OpConversionPattern< Op >
    {
        using parent_t = mlir::OpConversionPattern< Op >;
        using operation_t = Op;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
                operation_t op,
                typename operation_t::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            return Impl< operation_t >(op, ops, rewriter).convert();
        }
    };

} // namespace vast::util
