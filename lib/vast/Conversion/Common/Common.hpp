// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Conversion/Common/Patterns.hpp"

namespace vast::conv::irstollvm
{
    // I would consider to just use the entire namespace, everything
    // has (unfortunately) prefixed name with `LLVM` anyway.
    namespace LLVM = mlir::LLVM;
    using TypeConverter = util::tc::LLVMTypeConverter;

    template< typename op_t >
    struct base_pattern : operation_to_llvm_conversion_pattern< op_t >
    {
        using base = operation_to_llvm_conversion_pattern< op_t >;
        using TC_t = util::TypeConverterWrapper< TypeConverter >;

        TypeConverter &tc;

        base_pattern(TypeConverter &tc_) : base(tc_), tc(tc_) {}
        TypeConverter &type_converter() const { return tc; }

        auto dl(auto op) const { return tc.getDataLayoutAnalysis()->getAtOrAbove(op); }

        auto mk_alloca(auto &rewriter, mlir::Type trg_type, auto loc) const
        {
                auto count = rewriter.template create< LLVM::ConstantOp >(
                        loc,
                        type_converter().convertType(rewriter.getIndexType()),
                        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

                return rewriter.template create< LLVM::AllocaOp >(
                        loc, trg_type, count, 0);
        }

        // Some operations want more fine-grained control, and we really just
        // want to set entire dialects as illegal.
        static void legalize(conversion_target &) {}

        auto convert(mlir::Type t) const -> mlir::Type
        {
            auto trg_type = tc.convert_type_to_type(t);
            VAST_CHECK(trg_type, "Failed to convert type: {0}", t);
            return *trg_type;
        }
    };

    template< typename src_t, typename trg_t >
    struct one_to_one : base_pattern< src_t >
    {
        using base = base_pattern< src_t >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                    src_t op, typename src_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto target_ty = this->type_converter().convert_type_to_type(op.getType());
            auto new_op = rewriter.create< trg_t >(op.getLoc(), *target_ty, ops.getOperands());
            rewriter.replaceOp(op, {new_op});
            return mlir::success();
        }
    };

    // Ignore `src_t` and instead just us its operands.
    template< typename src_t >
    struct ignore_pattern : base_pattern< src_t >
    {
        using base = base_pattern< src_t >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                    src_t op, typename src_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            rewriter.replaceOp(op, ops.getOperands());
            return mlir::success();
        }
    };

    template< typename src_t >
    struct erase_pattern : base_pattern< src_t >
    {
        using base = base_pattern< src_t >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                    src_t op, typename src_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            rewriter.eraseOp(op);
            return mlir::success();
        }

        static void legalize(auto &trg)
        {
            trg.template addIllegalOp< src_t >();
        }
    };

    template< typename Op >
    bool has_llvm_only_types(Op op)
    {
        return vast::util::for_each_subtype(op.getResultTypes(), mlir::LLVM::isCompatibleType);
    }

    template < typename Op >
    bool has_llvm_return_type(Op op)
    {
        return mlir::LLVM::isCompatibleType(op.getResult().getType());
    }

} // namespace vast::conv::irstollvm
