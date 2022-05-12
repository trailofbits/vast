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
#include "vast/Util/Symbols.hpp"

#include <iostream>

namespace vast::hl
{
    using mctx_t = mlir::MLIRContext;

    namespace
    {
        std::size_t size(mlir::Region &region)
        {
            return std::distance(region.begin(), region.end());
        }

        std::size_t size(mlir::Block &block)
        {
            return std::distance(block.begin(), block.end());
        }

    }

    // TODO(lukas): In non-debug mode return `mlir::failure()` and do not log
    //              anything.
    #define VAST_PATTERN_CHECK(cond, fmt, ...) \
        VAST_CHECK(cond, fmt, __VA_ARGS__)

    namespace pattern
    {
        // NOTE(lukas): I would consider to just use the entire namespace, everything
        //              has (unfortunately) prefixed name with `LLVM` anyway.
        namespace LLVM = mlir::LLVM;

        // Since information is hidden in attribute, entire op must be an argument.
        bool is_variadic(mlir::FuncOp op)
        {
            // TODO(lukas): Implement once hl supports it.
            return false;
        }

        auto convert_fn_t(auto &tc, mlir::FuncOp op)
        -> std::tuple< mlir::TypeConverter::SignatureConversion, mlir::Type >
        {
            mlir::TypeConverter::SignatureConversion conversion(op.getNumArguments());
            auto target_type = tc.convertFunctionSignature(
                    op.getType(), is_variadic(op), conversion);
            return { std::move(conversion), target_type };
        }

        struct TypeConverter : mlir::LLVMTypeConverter, util::TCHelpers< TypeConverter >
        {
            using parent_t = mlir::LLVMTypeConverter;
            using helpers_t = util::TCHelpers< TypeConverter >;
            using maybe_type_t = typename util::TCHelpers< TypeConverter >::maybe_type_t;
            using self_t = TypeConverter;

            template< typename ... Args >
            TypeConverter(Args && ... args) : parent_t(std::forward< Args >(args) ... )
            {
                addConversion([&](hl::LValueType t) { return this->convert_lvalue_type(t); });
                addConversion([&](mlir::MemRefType t) { return this->convert_memref_type(t); });
                addConversion([&](mlir::UnrankedMemRefType t) {
                        return this->convert_memref_type(t);
                });
                // Overriding the inherited one to provide way to handle `hl.lvalue` in args.
                addConversion([&](mlir::FunctionType t) {
                        return this->convert_fn_t(t);
                });
            }

            maybe_types_t do_conversion(mlir::Type t)
            {
                types_t out;
                if (mlir::succeeded(this->convertTypes(t, out)))
                    return { std::move( out ) };
                return {};
            }

            auto make_ptr_type()
            {
                return [&](auto t)
                {
                    VAST_ASSERT(!t.template isa< mlir::NoneType >());
                    return mlir::LLVM::LLVMPointerType::get(t);
                };
            }

            maybe_type_t convert_lvalue_type(hl::LValueType t)
            {
                return Maybe(t.getElementType()).and_then(helpers_t::convert_type_to_type())
                                                .unwrap()
                                                .and_then(self_t::make_ptr_type())
                                                .take_wrapped< maybe_type_t >();
            }

            auto make_array(auto shape_)
            {
                return [shape = std::move(shape_)](auto t)
                {
                    mlir::Type out = LLVM::LLVMArrayType::get(t, shape.back());
                    for (int i = shape.size() - 2; i >= 0; --i)
                    {
                        out = LLVM::LLVMArrayType::get(out, shape[i]);
                    }
                    return out;
                };
            }

            maybe_type_t convert_memref_type(mlir::MemRefType t)
            {
                return Maybe(t.getElementType()).and_then(helpers_t::convert_type_to_type())
                                                .unwrap()
                                                .and_then(make_array(t.getShape()))
                                                .take_wrapped< maybe_type_t >();
            }

            maybe_type_t convert_memref_type(mlir::UnrankedMemRefType t)
            {
                VAST_UNIMPLEMENTED;
            }

            using signature_conversion_t = mlir::TypeConverter::SignatureConversion;
            using maybe_signature_conversion_t = std::optional< signature_conversion_t >;

            maybe_signature_conversion_t get_conversion_signature(mlir::FuncOp fn,
                                                                  bool variadic)
            {
                signature_conversion_t conversion(fn.getNumArguments());
                for (auto arg : llvm::enumerate(fn.getType().getInputs()))
                {
                    auto cty = convert_arg_t(arg.value());
                    if (!cty)
                        return {};
                    conversion.addInputs(arg.index(), *cty);
                }
                return { std::move(conversion) };
            }

            maybe_type_t convert_fn_t(mlir::FunctionType t)
            {
                auto a_res = this->on_types(t.getInputs(), &TypeConverter::convert_arg_t);
                auto r_res = this->on_types(t.getResults(), &TypeConverter::convert_ret_t);

                if (!a_res || !r_res)
                    return {};

                // LLVM function can have only one return value;
                if (r_res->size() != 1)
                    return {};
                // TODO(lukas): Not sure how to get info if the type is variadic or not here.
                return mlir::LLVM::LLVMFunctionType::get(r_res->front(), *a_res, false);
            }

            maybe_types_t on_types(auto range, auto fn)
            {
                types_t out;
                auto append = appender(out);

                for (auto t : range)
                    if (auto cty = (this->*fn)(t))
                        append(std::move(*cty));
                    else
                        return {};
                return { std::move(out) };
            }

            maybe_types_t convert_arg_t(mlir::Type t)
            {
                if (auto lvalue = t.dyn_cast< hl::LValueType >())
                    return this->convert_type_to_types(lvalue.getElementType());
                return this->convert_type_to_types(t);
            }

            maybe_types_t convert_ret_t(mlir::Type t)
            {
                if (auto lvalue = t.dyn_cast< hl::LValueType >())
                    return this->convert_type_to_types(lvalue.getElementType());
                return this->convert_type_to_types(t);
            }
        };

        template< typename O >
        struct BasePattern : mlir::ConvertOpToLLVMPattern< O >
        {
            using Base = mlir::ConvertOpToLLVMPattern< O >;
            using TC_t = util::TypeConverterWrapper< TypeConverter >;

            TypeConverter &tc;

            BasePattern(TypeConverter &tc_) : Base(tc_), tc(tc_) {}
            TypeConverter &type_converter() const { return tc; }
        };

        struct l_func_op : BasePattern< mlir::FuncOp >
        {
            using Base = BasePattern< mlir::FuncOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    mlir::FuncOp func_op, mlir::FuncOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto &tc = this->type_converter();
                auto maybe_target_type = tc.convert_fn_t(func_op.getType());
                auto maybe_signature = tc.get_conversion_signature(func_op,
                                                                   is_variadic(func_op));
                // Type converter failed.
                if (!maybe_target_type || !*maybe_target_type || !maybe_signature)
                    return mlir::failure();

                auto target_type = *maybe_target_type;
                auto signature = *maybe_signature;

                // TODO(lukas): We will want to lower a lot of stuff most likely.
                //              Copy those we want to preserve.
                mlir::SmallVector< mlir::NamedAttribute, 8 > new_attrs;

                if (auto original_arg_attr = func_op.getAllArgAttrs())
                {
                    mlir::SmallVector< mlir::Attribute, 8 > new_arg_attrs;
                    for (std::size_t i = 0; i < func_op.getNumArguments(); ++i)
                    {
                        const auto &mapping = signature.getInputMapping(i);
                        for (std::size_t j = 0; j < mapping->size; ++j)
                            new_arg_attrs[mapping->inputNo + j] = original_arg_attr[i];
                    }
                    new_attrs.push_back(rewriter.getNamedAttr(
                                mlir::FunctionOpInterface::getArgDictAttrName(),
                                rewriter.getArrayAttr(new_arg_attrs)));
                }
                // TODO(lukas): Linkage?
                auto linkage = LLVM::Linkage::Internal;
                auto new_func = rewriter.create< LLVM::LLVMFuncOp >(
                        func_op.getLoc(), func_op.getName(), target_type,
                        linkage, false, new_attrs);
                rewriter.inlineRegionBefore(func_op.getBody(),
                                            new_func.getBody(), new_func.end());
                util::convert_region_types(func_op, new_func, signature);

                if (mlir::failed(args_to_allocas(new_func, rewriter)))
                    return mlir::failure();
                rewriter.eraseOp(func_op);
                return mlir::success();
            }

            mlir::LogicalResult args_to_allocas(
                    mlir::LLVM::LLVMFuncOp fn,
                    mlir::ConversionPatternRewriter &rewriter) const
            {
                // TODO(lukas): Missing support in hl.
                if (fn.isVarArg())
                    return mlir::failure();

                if (fn.empty())
                    return mlir::failure();

                auto &block = fn.front();
                if (!block.isEntryBlock())
                    return mlir::failure();

                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(&block);

                for (auto arg : block.getArguments())
                    if (mlir::failed(arg_to_alloca(arg, block, rewriter)))
                        return mlir::failure();

                return mlir::success();
            }

            // TODO(lukas): Extract common codebase (there will be other places
            //              that need to create allocas).
            mlir::LogicalResult arg_to_alloca(mlir::BlockArgument arg, mlir::Block &block,
                                              mlir::ConversionPatternRewriter &rewriter) const
            {
                auto ptr_type = mlir::LLVM::LLVMPointerType::get(arg.getType());
                if (!ptr_type)
                    return mlir::failure();

                auto count = rewriter.create< LLVM::ConstantOp >(
                        arg.getLoc(),
                        type_converter().convertType(rewriter.getIndexType()),
                        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

                auto alloca = rewriter.create< LLVM::AllocaOp >(
                        arg.getLoc(), ptr_type, count, 0);

                arg.replaceAllUsesWith(alloca);
                rewriter.create< mlir::LLVM::StoreOp >(arg.getLoc(), arg, alloca);

                return mlir::success();
            }
        };

        struct l_constant_int : BasePattern< hl::ConstantIntOp >
        {
            using Base = BasePattern< hl::ConstantIntOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    hl::ConstantIntOp op, hl::ConstantIntOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.replaceOp(op, {make_from(op, rewriter, this->type_converter())});
                return mlir::success();
            }

            static LLVM::ConstantOp make_from(
                    hl::ConstantIntOp op,
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
                    hl::ReturnOp ret_op, hl::ReturnOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.create< LLVM::ReturnOp >(ret_op.getLoc(), ops.getOperands()[0]);
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
            VAST_ASSERT(size(src.initializer()) == 1);
            auto &init_region = src.initializer();
            auto &init_block = init_region.back();

            auto terminator = mlir::dyn_cast< T >(init_block.getTerminator());
            VAST_ASSERT(size(init_region) == 1 && terminator);
            rewriter.inlineRegionBefore(init_region, src->getBlock());
            auto ip = std::next(mlir::Block::iterator(src));
            VAST_ASSERT(ip != src->getBlock()->end());

            rewriter.mergeBlockBefore(&init_block, &*ip);
            return terminator;
        }

        struct l_var : BasePattern< hl::VarDecl >
        {
            using Base = BasePattern< hl::VarDecl >;
            using O = hl::VarDecl;
            using Base::Base;


            mlir::LogicalResult unfold_init(LLVM::AllocaOp alloca, hl::InitListExpr init,
                                            auto &rewriter) const
            {
                std::size_t i = 0;

                auto p_type = alloca.getType().cast< LLVM::LLVMPointerType >();
                VAST_PATTERN_CHECK(p_type, "Expected pointer.");
                auto a_type = p_type.getElementType().dyn_cast< LLVM::LLVMArrayType >();
                VAST_PATTERN_CHECK(a_type, "Expected array.");

                for (auto op : init.elements())
                {
                    auto e_type = LLVM::LLVMPointerType::get(a_type.getElementType());

                    auto index = rewriter.template create< LLVM::ConstantOp >(
                            op.getLoc(), type_converter().convertType(rewriter.getIndexType()),
                            rewriter.getIntegerAttr(rewriter.getIndexType(), i));
                    auto where = rewriter.template create< LLVM::GEPOp >(
                            alloca.getLoc(), e_type, alloca, index.getResult());
                    rewriter.template create< LLVM::StoreOp >(alloca.getLoc(), op, where);
                    ++i;
                }
                rewriter.eraseOp(init);
                return mlir::success();
            }

            mlir::LogicalResult make_init(LLVM::AllocaOp alloca, hl::ValueYieldOp yield,
                                          auto &rewriter) const
            {
                mlir::Value v = yield.getOperand();
                if (auto init_list = v.getDefiningOp< hl::InitListExpr >())
                    return unfold_init(alloca, init_list, rewriter);

                rewriter.template create< LLVM::StoreOp >(alloca.getLoc(), v, alloca);
                return mlir::success();
            }

            mlir::LogicalResult matchAndRewrite(
                    hl::VarDecl var_op, hl::VarDecl::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto ptr_type = type_converter().convertType(var_op.getType());
                if (!ptr_type)
                    return mlir::failure();

                auto count = rewriter.create< LLVM::ConstantOp >(
                        var_op.getLoc(),
                        type_converter().convertType(rewriter.getIndexType()),
                        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
                auto alloca = rewriter.create< LLVM::AllocaOp >(
                        var_op.getLoc(), ptr_type, count, 0);

                auto yield = inline_init_region< hl::ValueYieldOp >(var_op, rewriter);
                rewriter.setInsertionPoint(yield);
                if (!mlir::succeeded(make_init(alloca, yield, rewriter)))
                    return mlir::failure();

                rewriter.eraseOp(yield);
                rewriter.replaceOp(var_op, {alloca});

                return mlir::success();
            }


        };

        struct l_implicit_cast : BasePattern< hl::ImplicitCastOp >
        {
            using Base = BasePattern< hl::ImplicitCastOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        hl::ImplicitCastOp op, hl::ImplicitCastOp::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                if (op.kind() == hl::CastKind::LValueToRValue)
                {
                    auto loaded = rewriter.create< LLVM::LoadOp >(op.getLoc(), ops.getOperands()[0]);
                    rewriter.replaceOp(op, {loaded});
                    return mlir::success();
                }
                return mlir::failure();
            }

        };

        template< typename Src, typename Trg >
        struct one_to_one : BasePattern< Src >
        {
            using Base = BasePattern< Src >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        Src op, typename Src::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto trg_ty = this->type_converter().convert_type_to_type(op.getType());
                auto new_op = rewriter.create< Trg >(op.getLoc(), *trg_ty, ops.getOperands());
                rewriter.replaceOp(op, {new_op});
                return mlir::success();
            }
        };

        using l_add = one_to_one< hl::AddIOp, LLVM::AddOp >;
        using l_sub = one_to_one< hl::SubIOp, LLVM::SubOp >;

        template< typename Src, typename Trg >
        struct assign_pattern : BasePattern< Src >
        {
            using Base = BasePattern< Src >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        Src op, typename Src::Adaptor ops_,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto ops = ops_.getOperands();
                auto alloca = ops[0];

                std::vector< mlir::Value > m_ops{ ops.begin(), ops.end() };
                m_ops[0] = rewriter.create< LLVM::LoadOp >(op.getLoc(), ops[0]);
                auto trg_ty = this->type_converter().convert_type_to_type(op.src().getType());
                auto new_op = rewriter.create< Trg >(op.getLoc(), *trg_ty, m_ops);

                rewriter.create< LLVM::StoreOp >(op.getLoc(), new_op, alloca);

                // Assigns do not return a value.
                rewriter.eraseOp(op);
                return mlir::success();
            }
        };

        using l_assign_add = assign_pattern< hl::AddIAssignOp, LLVM::AddOp >;
        using l_assign_sub = assign_pattern< hl::SubIAssignOp, LLVM::SubOp >;

        template< typename Src >
        struct ignore_pattern : BasePattern< Src >
        {
            using Base = BasePattern< Src >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        Src op, typename Src::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.replaceOp(op, ops.getOperands());
                return mlir::success();
            }
        };

        using l_declref = ignore_pattern< hl::DeclRefOp >;

        struct l_call : BasePattern< hl::CallOp >
        {
            using Base = BasePattern< CallOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        hl::CallOp op, typename CallOp::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto module = op->getParentOfType< mlir::ModuleOp >();
                if (!module)
                    return mlir::failure();

                auto callee = module.lookupSymbol< mlir::LLVM::LLVMFuncOp >(op.callee());
                if (!callee)
                    return mlir::failure();

                auto rtys = this->type_converter().convert_types_to_types(
                        callee.getResultTypes());
                if (!rtys)
                    return mlir::failure();

                auto new_call = rewriter.create< mlir::LLVM::CallOp >(
                    op.getLoc(),
                    *rtys,
                    op.callee(),
                    ops.getOperands());
                rewriter.replaceOp(op, new_call.getResults());
                return mlir::success();
            }
        };

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
        llvm_options.useBarePtrCallConv = true;
        pattern::TypeConverter type_converter(&mctx, llvm_options , &dl_analysis);

        mlir::RewritePatternSet patterns(&mctx);
        patterns.add< pattern::l_func_op >(type_converter);
        patterns.add< pattern::l_var >(type_converter);
        patterns.add< pattern::l_constant_int >(type_converter);
        patterns.add< pattern::l_return >(type_converter);
        patterns.add< pattern::l_add >(type_converter);
        patterns.add< pattern::l_sub >(type_converter);
        patterns.add< pattern::l_declref >(type_converter);
        patterns.add< pattern::l_assign_add >(type_converter);
        patterns.add< pattern::l_assign_sub >(type_converter);
        patterns.add< pattern::l_implicit_cast >(type_converter);
        patterns.add< pattern::l_call >(type_converter);
        if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
            return signalPassFailure();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHLToLLPass()
{
    return std::make_unique< LowerHLToLLPass >();
}
