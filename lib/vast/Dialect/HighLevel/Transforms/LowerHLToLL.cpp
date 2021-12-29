// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/TypeConverter.hpp"

#include <iostream>

namespace vast::hl
{
    using type_converter_t = mlir::LLVMTypeConverter;
    using mctx_t = mlir::MLIRContext;

    std::size_t size(mlir::Region &region)
    {
        return std::distance(region.begin(), region.end());
    }

    std::size_t size(mlir::Block &block)
    {
        return std::distance(block.begin(), block.end());
    }

    namespace pattern
    {
        // NOTE(lukas): I would consider to just use the entire namespace, everything
        //              has (unfortunately) prefixed name with `LLVM` anyway.
        namespace LLVM = mlir::LLVM;

        using types_t = mlir::SmallVector< mlir::Type, 4 >;
        using maybe_types_t = std::optional< types_t >;

        using maybe_type_t = Maybe< mlir::Type >;

        maybe_types_t convert_types(auto &type_converter, const auto &types)
        {
            types_t out;
            if (mlir::succeeded(type_converter.convertTypes(types, out)))
                return { std::move( out ) };
            return {};
        }

        maybe_types_t convert_type_attr(auto &tc, mlir::Operation *op)
        {
            auto type_attr = op->template getAttrOfType< mlir::TypeAttr >("type");
            if (!type_attr)
                return {};

            return convert_types(tc, llvm::makeArrayRef(type_attr.getValue()));
        }

        // Since information is hidden in attribute, entire op must be an argument.
        bool is_variadic(mlir::FuncOp op)
        {
            // TODO(lukas): Implement once hl supports it.
            return false;
        }

        auto convert_fn_t(type_converter_t &tc, mlir::FuncOp op)
        -> std::tuple< mlir::TypeConverter::SignatureConversion, mlir::Type >
        {
            mlir::TypeConverter::SignatureConversion conversion(op.getNumArguments());
            auto target_type = tc.convertFunctionSignature(
                    op.getType(), is_variadic(op), conversion);
            return { std::move(conversion), target_type };
        }

        template< typename O >
        struct BasePattern : mlir::ConvertOpToLLVMPattern< O >
        {
            using Base = mlir::ConvertOpToLLVMPattern< O >;
            using Base::Base;

            using TC_t = TypeConverterWrapper< mlir::LLVMTypeConverter >;

            TC_t type_converter() const { return TC_t{ *this->getTypeConverter() }; }
        };

        struct l_func_op : BasePattern< mlir::FuncOp >
        {
            using Base = BasePattern< mlir::FuncOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    mlir::FuncOp func_op, mlir::ArrayRef< mlir::Value > ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto tc = this->getTypeConverter();
                auto [conversion, target_type] = convert_fn_t(*tc, func_op);
                // Type converter failed.
                if (!target_type)
                    return mlir::failure();

                // TODO(lukas): We will want to lower a lot of stuff most likely.
                //              Copy those we want to preserve.
                mlir::SmallVector< mlir::NamedAttribute, 8 > new_attrs;

                if (auto original_arg_attr = func_op.getAllArgAttrs())
                {
                    mlir::SmallVector< mlir::Attribute, 8 > new_arg_attrs;
                    for (std::size_t i = 0; i < func_op.getNumArguments(); ++i)
                    {
                        const auto &mapping = conversion.getInputMapping(i);
                        for (std::size_t j = 0; j < mapping->size; ++j)
                            new_arg_attrs[mapping->inputNo + j] = original_arg_attr[i];
                    }
                    new_attrs.push_back(rewriter.getNamedAttr(
                                mlir::function_like_impl::getArgDictAttrName(),
                                rewriter.getArrayAttr(new_arg_attrs)));
                }
                // TODO(lukas): Linkage?
                auto linkage = LLVM::Linkage::Internal;
                auto new_func = rewriter.create< LLVM::LLVMFuncOp >(
                        func_op.getLoc(), func_op.getName(), target_type,
                        linkage, false, new_attrs);
                rewriter.inlineRegionBefore(func_op.getBody(),
                                            new_func.getBody(), new_func.end());
                if (mlir::failed(rewriter.convertRegionTypes(&new_func.getBody(),
                                                             *tc, &conversion)))
                    return mlir::failure();
                rewriter.eraseOp(func_op);
                return mlir::success();

            }
        };

        struct l_constant_int : BasePattern< hl::ConstantIntOp >
        {
            using Base = BasePattern< hl::ConstantIntOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    hl::ConstantIntOp op, mlir::ArrayRef< mlir::Value > ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.replaceOp(op, {make_from(op, ops, rewriter, this->type_converter())});
                return mlir::success();
            }

            static LLVM::ConstantOp make_from(
                    hl::ConstantIntOp op,
                    mlir::ArrayRef< mlir::Value > ops,
                    mlir::ConversionPatternRewriter &rewriter,
                    auto &&tc)
            {
                auto src_t = op.getType();
                auto trg_t = tc.convert_type_to_type(src_t);

                return rewriter.create< LLVM::ConstantOp >(op.getLoc(), *trg_t,
                                                           op.getValue());
            }
        };

        struct l_return : BasePattern< hl::ReturnOp >
        {
            using Base = BasePattern< hl::ReturnOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    hl::ReturnOp ret_op, mlir::ArrayRef< mlir::Value > ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.create< LLVM::ReturnOp >(ret_op.getLoc(), ops[0]);
                rewriter.eraseOp(ret_op);
                return mlir::success();
            }
        };

        // Inline the region that is responsible for initialization
        //  * `rewriter` insert point is invalidated (although documentation of called
        //    methods does not state it, experimentally it is corrupted)
        //  * terminator is returned to be used & erased by caller.
        template< typename T >
        T inline_init_region(auto src, auto &rewriter)
        {
            ASSERT(size(src.initializer()) == 1);
            auto &init_region = src.initializer();
            auto &init_block = init_region.back();

            auto terminator = mlir::dyn_cast< T >(init_block.getTerminator());
            ASSERT(size(init_region) == 1 && terminator);
            rewriter.inlineRegionBefore(init_region, src->getBlock());
            auto ip = std::next(mlir::Block::iterator(src));
            ASSERT(ip != src->getBlock()->end());

            rewriter.mergeBlockBefore(&init_block, &*ip);
            return terminator;
        }


        struct l_var : BasePattern< hl::VarOp >
        {
            using Base = BasePattern< hl::VarOp >;
            using O = hl::VarOp;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    hl::VarOp var_op, mlir::ArrayRef< mlir::Value > ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto ty = convert_type_attr(*this->getTypeConverter(), var_op);
                if (!ty || ty->size() > 1)
                    return mlir::failure();

                auto wrap_type = LLVM::LLVMPointerType::get(*(ty->begin()));
                auto count = rewriter.create< LLVM::ConstantOp >(
                        var_op.getLoc(),
                        this->getTypeConverter()->convertType(rewriter.getIndexType()),
                        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
                auto alloca = rewriter.create< LLVM::AllocaOp >(
                        var_op.getLoc(), wrap_type, count, 0);

                auto yield = inline_init_region< hl::ValueYieldOp >(var_op, rewriter);
                rewriter.setInsertionPoint(yield);
                rewriter.create< LLVM::StoreOp >(var_op.getLoc(), yield->getOperand(0), alloca);

                rewriter.eraseOp(yield);
                rewriter.eraseOp(var_op);
                return mlir::success();
            }
        };

        template< typename Src, typename Trg >
        struct one_to_one : BasePattern< Src >
        {
            using Base = BasePattern< Src >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        Src op, mlir::ArrayRef< mlir::Value > ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto trg_ty = this->type_converter().convert_type_to_type(op.getType());
                auto new_op = rewriter.create< Trg >(op.getLoc(), *trg_ty, ops);
                rewriter.replaceOp(op, {new_op});
                return mlir::success();
            }
        };

        using l_add = one_to_one< hl::AddIOp, LLVM::AddOp >;
        using l_sub = one_to_one< hl::SubIOp, LLVM::SubOp >;

    } // namespace pattern


    struct LowerHLToLLPass : LowerHLToLLBase< LowerHLToLLPass >
    {
        void runOnOperation() override;
    };

    void LowerHLToLLPass::runOnOperation()
    {
        auto &mctx = this->getContext();
        mlir::ModuleOp op = this->getOperation();


        mlir::ConversionTarget trg(mctx);
        trg.addLegalDialect< mlir::LLVM::LLVMDialect >();
        trg.addLegalOp< mlir::ModuleOp >();

        const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();

        mlir::LowerToLLVMOptions llvm_options{ &mctx };
        mlir::LLVMTypeConverter type_converter(&mctx, llvm_options , &dl_analysis);

        mlir::RewritePatternSet patterns(&mctx);
        patterns.add< pattern::l_func_op >(type_converter);
        patterns.add< pattern::l_var >(type_converter);
        patterns.add< pattern::l_constant_int >(type_converter);
        patterns.add< pattern::l_return >(type_converter);
        patterns.add< pattern::l_add >(type_converter);
        patterns.add< pattern::l_sub >(type_converter);
        if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
            return signalPassFailure();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHLToLLPass()
{
    return std::make_unique< LowerHLToLLPass >();
}
