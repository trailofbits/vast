// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/Common.hpp"

#include "vast/Conversion/Common/Types.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
VAST_UNRELAX_WARNINGS

namespace vast {

    namespace detail
    {
        static inline bool is_ok( const auto &v ) { return static_cast< bool >( v ); }
        static inline bool is_ok( logical_result r ) { return mlir::succeeded( r ); }
    } // namespace detail

    // TODO(conv): In non-debug mode return `mlir::failure()` and do not log
    //             anything.
    // If this header is ever exported, probably remove this.
    #define VAST_PATTERN_CHECK(cond, ...) \
        VAST_CHECK( detail::is_ok(cond), __VA_ARGS__)

    #define VAST_PATTERN_FAIL(...) \
        VAST_UNREACHABLE(__VA_ARGS__)


    template< typename T >
    concept has_legalize = requires ( T a )
    {
        a.legalize( std::declval< conversion_target & >() );
    };

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

        static void legalize( conversion_target &trg )
        {
            trg.addIllegalOp< op_t >();
        }
    };

    struct llvm_pattern_utils
    {
        auto iN(auto &rewriter, auto loc, mlir::Type type, auto val) const
        {
            return rewriter.template create< mlir::LLVM::ConstantOp >(
                    loc,
                    type,
                    rewriter.getIntegerAttr(type, val));
        }

        auto fN(auto &rewriter, auto loc, mlir::Type type, auto val) const
        {
            return rewriter.template create< mlir::LLVM::ConstantOp >(
                    loc,
                    type,
                    rewriter.getFloatAttr(type, val));
        }

        auto constant(auto &rewriter, auto loc, mlir::Type type, auto val) const
        {
            if (type.isIntOrIndex())
                return iN(rewriter, loc, type, val);
            if (type.isa< mlir::FloatType >())
                return fN(rewriter, loc, type, val);
            VAST_UNREACHABLE("not an integer or float type");
        }

        static auto create_trunc_or_sext(auto op, mlir::Type target, auto &rewriter,
                                         mlir::Location loc, const auto &dl)
            -> mlir::Value
        {
            VAST_ASSERT(op.getType().template isa< mlir::IntegerType >() &&
                        target.isa< mlir::IntegerType >());
            auto new_bw = dl.getTypeSizeInBits(target);
            auto original_bw = dl.getTypeSizeInBits(op.getType());

            if (new_bw == original_bw)
                return op;
            else if (new_bw > original_bw)
                return rewriter.template create< mlir::LLVM::SExtOp >(loc, target, op);
            else
                return rewriter.template create< mlir::LLVM::TruncOp >(loc, target, op);
        }
    };

    template< typename op_t >
    struct operation_to_llvm_conversion_pattern
        : mlir::ConvertOpToLLVMPattern< op_t >, llvm_pattern_utils
    {
            using base = mlir::ConvertOpToLLVMPattern< op_t >;
            using base::base;

            using llvm_util = llvm_pattern_utils;
            using llvm_util::llvm_util;


            static void legalize( conversion_target &trg )
            {
                trg.addIllegalOp< op_t >();
            }
    };

} // namespace vast
