// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

namespace vast {

    using pattern_rewriter = mlir::PatternRewriter;

    using conversion_rewriter = mlir::ConversionPatternRewriter;

    template< typename derived_pattern >
    struct mlir_pattern_mixin {

        using integer_attr_t = mlir::IntegerAttr;

        derived_pattern& self() { return static_cast< derived_pattern& >(this); }
        const derived_pattern& self() const { return static_cast< const derived_pattern& >(this); }

        template< typename T >
        mlir_type bitwidth_type() const {
            auto ctx = self().getContext();
            return mlir::IntegerType::get(ctx, bits< T >());
        }

        template< typename T >
        integer_attr_t interger_attr(T v) const { return integer_attr_t::get(bitwidth_type< T >(), v); }

        integer_attr_t  u8(uint8_t  v) const { return interger_attr(v); }
        integer_attr_t u16(uint16_t v) const { return interger_attr(v); }
        integer_attr_t u32(uint32_t v) const { return interger_attr(v); }
        integer_attr_t u64(uint64_t v) const { return interger_attr(v); }

        integer_attr_t  i8(int8_t  v) const { return interger_attr(v); }
        integer_attr_t i16(int16_t v) const { return interger_attr(v); }
        integer_attr_t i32(int32_t v) const { return interger_attr(v); }
        integer_attr_t i64(int64_t v) const { return interger_attr(v); }
    };

    template< typename op_t >
    struct operation_rewrite_pattern
        : mlir_pattern_mixin< operation_rewrite_pattern< op_t > >
        , mlir::OpRewritePattern< op_t >
    {
        using base = mlir::OpRewritePattern< op_t >;
        using base::base;
    };

    template< typename op_t >
    struct operation_conversion_pattern
        : mlir_pattern_mixin< operation_conversion_pattern< op_t > >
        , mlir::OpConversionPattern< op_t >
    {
        using base = mlir::OpConversionPattern< op_t >;
        using base::base;
    };

} // namespace vast
