// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast
{
    using Rewriter = mlir::ConversionPatternRewriter;

    template< typename Op >
    using OpConversionPattern = mlir::OpConversionPattern< Op >;

} // namespace vast

namespace vast::util
{
    template< typename Op >
    struct State
    {
        Op op;
        typename Op::Adaptor operands;
        Rewriter &rewriter;

        State(Op op, typename Op::Adaptor operands, Rewriter &rewriter)
            : op(op), operands(operands), rewriter(rewriter)
        {}

        static mlir::Block &solo_block(mlir::Region &region)
        {
            VAST_ASSERT(region.hasOneBlock());
            return *region.begin();
        }
    };

    template< typename Op, template< typename > class Impl >
    struct BasePattern : OpConversionPattern< Op >
    {
        using parent_t = mlir::OpConversionPattern< Op >;
        using operation_t = Op;
        using parent_t::parent_t;

        mlir::LogicalResult matchAndRewrite(
                operation_t op,
                typename operation_t::Adaptor ops,
                Rewriter &rewriter) const override
        {
            // Required, because even if the method is overloaded in one of the
            // children, this method must still compile.
            if constexpr( std::is_constructible_v< Impl< operation_t >,
                    decltype(op),
                    decltype(ops),
                    decltype(rewriter) >)
            {
                return Impl< operation_t >(op, ops, rewriter).convert();
            } else {
                return mlir::failure();
            }
        }
    };

    // TODO(lukas) Unfortunately, extending the class is rather tedious, might
    //             need to rethink how pieces fit together.
    template< typename Op, typename TC, template< typename > class Impl >
    struct TypeConvertingPattern : BasePattern< Op, Impl >
    {
        using parent_t = BasePattern< Op, Impl >;
        using operation_t = typename parent_t::operation_t;

        TC &tc;

        TypeConvertingPattern(TC &tc, mcontext_t *mctx)
            : parent_t(mctx), tc(tc)
        {}

        mlir::LogicalResult matchAndRewrite(
                operation_t op,
                typename operation_t::Adaptor ops,
                Rewriter &rewriter) const override
        {
            return Impl< operation_t >(tc, op, ops, rewriter).convert();
        }

    };

} // namespace vast::util
