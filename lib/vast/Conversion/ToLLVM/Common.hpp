// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/TypeList.hpp"

#include "vast/Util/TypeUtils.hpp"

#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/TypeConverters/LLVMTypeConverter.hpp"

namespace vast::conv::irstollvm {
    // I would consider to just use the entire namespace, everything
    // has (unfortunately) prefixed name with `LLVM` anyway.
    namespace LLVM = mlir::LLVM;

    template< typename op_t >
    struct base_pattern : operation_to_llvm_conversion_pattern< op_t >
    {
        using base = operation_to_llvm_conversion_pattern< op_t >;

        tc::FullLLVMTypeConverter &tc;

        base_pattern(tc::FullLLVMTypeConverter &tc_) : base(tc_), tc(tc_) {}

        tc::FullLLVMTypeConverter &type_converter() const { return tc; }

        auto dl(auto op) const { return tc.getDataLayoutAnalysis()->getAtOrAbove(op); }

        auto mk_alloca(auto &rewriter, mlir_type trg_type, auto loc) const {
            auto count = rewriter.template create< LLVM::ConstantOp >(
                loc, type_converter().convertType(rewriter.getIndexType()),
                rewriter.getIntegerAttr(rewriter.getIndexType(), 1)
            );

            return rewriter.template create< LLVM::AllocaOp >(loc, trg_type, count, 0);
        }

        // Some operations want more fine-grained control, and we really just
        // want to set entire dialects as illegal.
        static void legalize(conversion_target &) {}

        auto convert(mlir_type t) const -> mlir_type {
            auto trg_type = tc.convert_type_to_type(t);
            VAST_CHECK(trg_type, "Failed to convert type: {0}", t);
            return *trg_type;
        }

        auto mk_index(auto loc, std::size_t idx, auto &rewriter) const
            -> mlir::LLVM::ConstantOp {
            auto index_type = convert(rewriter.getIndexType());
            return rewriter.template create< mlir::LLVM::ConstantOp >(
                loc, index_type, rewriter.getIntegerAttr(index_type, idx)
            );
        }

        auto undef(auto &rewriter, auto loc, auto type) const {
            return rewriter.template create< mlir::LLVM::UndefOp >(loc, type);
        }
    };

    template< typename src_t, typename trg_t >
    struct one_to_one : base_pattern< src_t >
    {
        using base = base_pattern< src_t >;
        using base::base;

        using adaptor_t = typename src_t::Adaptor;

        logical_result
        matchAndRewrite(src_t op, adaptor_t ops, conversion_rewriter &rewriter) const override {
            auto target_ty = this->type_converter().convert_type_to_type(op.getType());
            auto new_op = rewriter.create< trg_t >(op.getLoc(), *target_ty, ops.getOperands());
            rewriter.replaceOp(op, new_op);
            return mlir::success();
        }
    };

    // Ignore `src_t` and instead just us its operands.
    template< typename src_t >
    struct ignore_pattern : base_pattern< src_t >
    {
        using base = base_pattern< src_t >;
        using base::base;

        using adaptor_t = typename src_t::Adaptor;

        logical_result
        matchAndRewrite(src_t op, adaptor_t ops, conversion_rewriter &rewriter) const override {
            rewriter.replaceOp(op, ops.getOperands());
            return mlir::success();
        }
    };

    template< typename src_t >
    struct erase_pattern : base_pattern< src_t >
    {
        using base = base_pattern< src_t >;
        using base::base;

        using adaptor_t = typename src_t::Adaptor;

        logical_result
        matchAndRewrite(src_t op, adaptor_t ops, conversion_rewriter &rewriter) const override {
            rewriter.eraseOp(op);
            return mlir::success();
        }

        static void legalize(auto &trg) { trg.template addIllegalOp< src_t >(); }
    };

    static auto get_is_illegal(auto &tc) {
        return [&](mlir_type type) { return !tc.isLegal(type); };
    }

    template< typename T, typename type_converter >
    auto get_has_only_legal_types(type_converter &tc) {
        return [&](T op) -> bool { return !has_type_somewhere(op, get_is_illegal(tc)); };
    }

    template< typename T, typename type_converter >
    auto get_has_legal_return_type(type_converter &tc) {
        return [&](T op) -> bool {
            return !contains_subtype(op.getResult().getType(), get_is_illegal(tc));
        };
    }

} // namespace vast::conv::irstollvm
