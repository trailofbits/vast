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
        VAST_CHECK( ::vast::detail::is_ok(cond), __VA_ARGS__)

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

    struct generic_conversion_pattern : mlir::ConversionPattern
    {
        using base = mlir::ConversionPattern;

        generic_conversion_pattern(mlir::TypeConverter &tc,
                                   mcontext_t &mctx)
            : base(tc, mlir::Pattern::MatchAnyOpTypeTag{}, 1, &mctx)
        {}
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
        mlir_value iN(auto &rewriter, auto loc, mlir_type type, auto val) const {
            return rewriter.template create< mlir::LLVM::ConstantOp >(
                loc, type, rewriter.getIntegerAttr(type, val)
            );
        }

        mlir_value fN(auto &rewriter, auto loc, mlir_type type, auto val) const {
            return rewriter.template create< mlir::LLVM::ConstantOp >(
                loc, type, rewriter.getFloatAttr(type, val)
            );
        }

        mlir_value null_ptr(auto &rewriter, auto loc, mlir_type type) const {
            return rewriter.template create< mlir::LLVM::NullOp >(loc, type);
        }

        mlir_value ptr(auto &rewriter, auto loc, mlir_type type, auto val) const {
            if (val) {
                auto ptr_val = rewriter.template create< mlir::LLVM::ConstantOp >(
                    loc, rewriter.getI64Type(), val
                );
                return rewriter.template create< mlir::LLVM::IntToPtrOp >(loc, type, ptr_val);
            } else {
                return null_ptr(rewriter, loc, type);
            }
        }

        mlir_value constant(auto &rewriter, auto loc, mlir_type type, auto val) const
        {
            if (type.isIntOrIndex())
                return iN(rewriter, loc, type, val);
            if (mlir::isa< mlir::FloatType >(type))
                return fN(rewriter, loc, type, val);
            if (mlir::isa< mlir::LLVM::LLVMPointerType >(type))
                return ptr(rewriter, loc, type, val);
            VAST_UNREACHABLE("not an integer or float type");
        }


        static auto replace_with_trunc_or_ext(
            auto op, auto src, auto orig_src_type, mlir_type dst_type, auto &rewriter
        ) -> mlir_value {
            auto src_type = src.getType();

            auto src_bw = src_type.getIntOrFloatBitWidth();
            auto dst_bw = dst_type.getIntOrFloatBitWidth();

            if (src_bw > dst_bw) {
                return rewriter.template replaceOpWithNewOp< mlir::LLVM::TruncOp >(op, dst_type, src);
            } else if (src_bw < dst_bw) {
                if (orig_src_type.isSignedInteger()) {
                    return rewriter.template replaceOpWithNewOp< mlir::LLVM::SExtOp >(op, dst_type, src);
                } else {
                    return rewriter.template replaceOpWithNewOp< mlir::LLVM::ZExtOp >(op, dst_type, src);
                }
            } else {
                // src_bw == dst_bw
                rewriter.replaceOp(op, src);
                return src;
            }
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

    template< typename op_t >
    struct match_and_rewrite_state_capture
    {
        using adaptor_t = typename op_t::Adaptor;

        op_t op;
        adaptor_t operands;
        conversion_rewriter &rewriter;
    };

} // namespace vast
