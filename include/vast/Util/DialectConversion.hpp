// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Types.hpp"
#include "vast/Util/Common.hpp"

namespace vast
{
    template< typename Op >
    using OpConversionPattern = mlir::OpConversionPattern< Op >;

} // namespace vast

namespace vast::util
{
    template< typename Op >
    struct State
    {
        using adaptor_t = typename Op::Adaptor;

        Op op;
        adaptor_t operands;
        conversion_rewriter &rewriter;

        State(Op op, adaptor_t operands, conversion_rewriter &rewriter)
            : op(op), operands(operands), rewriter(rewriter)
        {}

        static mlir::Block &solo_block(mlir::Region &region)
        {
            VAST_ASSERT(region.hasOneBlock());
            return *region.begin();
        }
    };

    template< typename operation_t, template< typename > class Impl >
    struct BasePattern : OpConversionPattern< operation_t >
    {
        using base = OpConversionPattern< operation_t >;
        using base::base;

        using adaptor_t = typename operation_t::Adaptor;

        mlir::LogicalResult matchAndRewrite(
            operation_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
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
    template< typename op_t, typename type_converter, template< typename > class Impl >
    struct TypeConvertingPattern : BasePattern< op_t, Impl >
    {
        using base = BasePattern< op_t, Impl >;
        using adaptor_t = typename op_t::Adaptor;

        type_converter &tc;

        TypeConvertingPattern(type_converter &tc, mcontext_t *mctx)
            : base(mctx), tc(tc)
        {}

        mlir::LogicalResult matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            return Impl< op_t >(tc, op, ops, rewriter).convert();
        }
    };

} // namespace vast::util
