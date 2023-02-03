// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/Common/Passes.hpp"

#include <iostream>

namespace vast
{
    // TODO(lukas): In non-debug mode return `mlir::failure()` and do not log
    //              anything.
    #define VAST_PATTERN_CHECK(cond, fmt, ...) \
        VAST_CHECK(cond, fmt, __VA_ARGS__)

    namespace pattern
    {
        // NOTE(lukas): I would consider to just use the entire namespace, everything
        //              has (unfortunately) prefixed name with `LLVM` anyway.
        namespace LLVM = mlir::LLVM;

        using TypeConverter = util::tc::LLVMTypeConverter;

        template< typename Op >
        struct BasePattern : mlir::ConvertOpToLLVMPattern< Op >
        {
            using Base = mlir::ConvertOpToLLVMPattern< Op >;
            using TC_t = util::TypeConverterWrapper< TypeConverter >;

            TypeConverter &tc;

            BasePattern(TypeConverter &tc_) : Base(tc_), tc(tc_) {}
            TypeConverter &type_converter() const { return tc; }


            auto mk_alloca(auto &rewriter, mlir::Type trg_type, auto loc) const
            {
                    auto count = rewriter.template create< LLVM::ConstantOp >(
                            loc,
                            type_converter().convertType(rewriter.getIndexType()),
                            rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

                    return rewriter.template create< LLVM::AllocaOp >(
                            loc, trg_type, count, 0);
            }

            auto iN(auto &rewriter, auto loc, mlir::Type type, auto val) const
            {
                return rewriter.template create< LLVM::ConstantOp >(
                        loc,
                        type,
                        rewriter.getIntegerAttr(type, val));
            }
        };

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

        using ignore_patterns = util::type_list<
            ignore_pattern< hl::DeclRefOp >
        >;

        template< typename Op >
        struct inline_region_from_op : BasePattern< Op >
        {
            using Base = BasePattern< Op >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    Op unit_op, typename Op::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto parent = unit_op.getBody().getParentRegion();
                rewriter.inlineRegionBefore(unit_op.getBody(), *parent, parent->end());

                // splice newly created translation unit block in the module
                auto &unit_block = parent->back();
                rewriter.mergeBlocks(&parent->front(), &unit_block, unit_block.getArguments());

                rewriter.eraseOp(unit_op);
                return mlir::success();
            }
        };

        using inline_region_from_op_conversions = util::type_list<
            inline_region_from_op< hl::TranslationUnitOp >,
            inline_region_from_op< hl::ScopeOp >
        >;


        struct uninit_var : BasePattern< ll::UninitializedVar >
        {
            using op_t = ll::UninitializedVar;
            using Base = BasePattern< op_t >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    op_t op, typename op_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto trg_type = tc.convert_type_to_type(op.getType());
                VAST_PATTERN_CHECK(trg_type, "Could not convert vardecl type");

                auto alloca = mk_alloca(rewriter, *trg_type, op.getLoc());
                rewriter.replaceOp(op, {alloca});

                return mlir::success();
            }
        };

        struct initialize_var : BasePattern< ll::InitializeVar >
        {
            using op_t = ll::InitializeVar;
            using Base = BasePattern< op_t >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    op_t op, typename op_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto index_type = tc.convert_type_to_type(rewriter.getIndexType());
                VAST_PATTERN_CHECK(index_type, "Was not able to convert index type");

                for (auto element : ops.elements())
                {
                    // TODO(lukas): This is not ideal - when lowering into ll we most
                    //              likely want to have multiple types of initializations?
                    //              For example `memset` or ctor call?
                    if (auto init_list_expr = element.getDefiningOp< hl::InitListExpr >())
                    {
                        std::size_t i = 0;
                        for (auto expr_elem : init_list_expr.getElements())
                        {
                            auto e_type = LLVM::LLVMPointerType::get(expr_elem.getType());
                            auto index = rewriter.create< LLVM::ConstantOp >(
                                    op.getLoc(), *index_type,
                                    rewriter.getIntegerAttr(rewriter.getIndexType(), i++));
                            auto gep = rewriter.create< LLVM::GEPOp >(
                                    op.getLoc(), e_type, ops.var(), index.getResult());

                            rewriter.create< LLVM::StoreOp >(op.getLoc(), expr_elem, gep);
                        }
                        rewriter.eraseOp(init_list_expr);
                        break;
                    }

                    rewriter.create< LLVM::StoreOp >(op.getLoc(), element, ops.var());
                }

                // While op is a value, there is no reason not to use the previous alloca,
                // since we just initialized it.
                rewriter.replaceOp(op, ops.var());

                return mlir::success();
            }
        };

        struct init_list_expr : BasePattern< hl::InitListExpr >
        {
            using op_t = hl::InitListExpr;
            using Base = BasePattern< op_t >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    op_t op, typename op_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                std::vector< mlir::Value > converted;
                // TODO(lukas): Can we just directly use `getElements`?
                for (auto element : ops.getElements())
                    converted.push_back(element);


                // We cannot replace the op with just `converted` as there is an internal
                // assert that number we replace the same count of things.
                VAST_PATTERN_CHECK(op.getNumResults() == 1, "Unexpected number of results");
                auto res_type = tc.convert_type_to_type(op.getType(0));
                VAST_PATTERN_CHECK(res_type, "Failed conversion of InitListExpr res type");
                auto new_op = rewriter.create< hl::InitListExpr >(
                        op.getLoc(), *res_type, converted);
                rewriter.replaceOp(op, new_op.getResults());

                return mlir::success();
            }
        };

        using init_conversions = util::type_list<
            uninit_var,
            initialize_var,
            init_list_expr
        >;

        struct func_op : BasePattern< mlir::func::FuncOp >
        {
            using Base = BasePattern< mlir::func::FuncOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    mlir::func::FuncOp func_op, mlir::func::FuncOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto &tc = this->type_converter();

                auto maybe_target_type = tc.convert_fn_t(func_op.getFunctionType());
                auto maybe_signature =
                    tc.get_conversion_signature(func_op, util::tc::is_variadic(func_op));

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
                auto linkage = LLVM::Linkage::External;
                auto new_func = rewriter.create< LLVM::LLVMFuncOp >(
                        func_op.getLoc(), func_op.getName(), target_type,
                        linkage, false, LLVM::CConv::C, new_attrs);
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

                auto alloca_op = rewriter.create< LLVM::AllocaOp >(
                        arg.getLoc(), ptr_type, count, 0);

                arg.replaceAllUsesWith(alloca_op);
                rewriter.create< mlir::LLVM::StoreOp >(arg.getLoc(), arg, alloca_op);

                return mlir::success();
            }
        };

        struct constant_int : BasePattern< hl::ConstantOp >
        {
            using Base = BasePattern< hl::ConstantOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    hl::ConstantOp op, hl::ConstantOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.replaceOp(op, {make_from(op, rewriter, this->type_converter())});
                return mlir::success();
            }

            mlir::Attribute convert_attr(auto attr, auto op,
                                         mlir::ConversionPatternRewriter &rewriter) const
            {
                auto target_type = this->type_converter().convert_type_to_type(attr.getType());
                const auto &dl = this->type_converter().getDataLayoutAnalysis()
                                                       ->getAtOrAbove(op);
                if (!target_type)
                    return {};

                if (auto float_attr = attr.template dyn_cast< hl::FloatAttr >())
                {
                    // NOTE(lukas): We cannot simply forward the return value of `getValue()`
                    //              because it can have different semantics than one expected
                    //              by `FloatAttr`.
                    // TODO(lukas): Is there a better way to convert this?
                    //              Ideally `APFloat -> APFloat`.
                    double raw_value = float_attr.getValue().convertToDouble();
                    return rewriter.getFloatAttr(*target_type, raw_value);
                }
                if (auto int_attr = attr.template dyn_cast< hl::IntegerAttr >())
                {
                    auto size = dl.getTypeSizeInBits(*target_type);
                    auto coerced = int_attr.getValue().sextOrTrunc(size);
                    return rewriter.getIntegerAttr(*target_type, coerced);
                }
                // Not implemented yet.
                return {};
            }

            LLVM::ConstantOp make_from(
                    hl::ConstantOp op,
                    mlir::ConversionPatternRewriter &rewriter,
                    auto &&tc) const
            {
                auto src_ty = op.getType();
                auto target_ty = tc.convert_type_to_type(src_ty);

                auto attr = convert_attr(op.getValue(), op, rewriter);
                return rewriter.create< LLVM::ConstantOp >(op.getLoc(), *target_ty, attr);
            }
        };

        struct ret : BasePattern< hl::ReturnOp >
        {
            using Base = BasePattern< hl::ReturnOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                    hl::ReturnOp ret_op, hl::ReturnOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.create< LLVM::ReturnOp >(ret_op.getLoc(), ops.getOperands());
                rewriter.eraseOp(ret_op);
                return mlir::success();
            }
        };

        // TODO(lukas): Move to some utils.
        auto create_trunc_or_sext(auto op, mlir::Type target, auto &rewriter,
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

        struct implicit_cast : BasePattern< hl::ImplicitCastOp >
        {
            using Base = BasePattern< hl::ImplicitCastOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        hl::ImplicitCastOp op, hl::ImplicitCastOp::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override {
                auto trg_type = tc.convert_type_to_type(op.getType());
                VAST_PATTERN_CHECK(trg_type, "Did not convert type");
                if (op.getKind() == hl::CastKind::LValueToRValue)
                {
                    // TODO(lukas): Without `--ccopts -xc` in case of `c = (x = 5)`
                    //              there will be a LValueToRValue cast on rvalue from
                    //              `(x = 5)` - not sure why that is so, so just fail
                    //              gracefully for now.
                    if (!op.getOperand().getType().isa< hl::LValueType >())
                        return mlir::failure();

                    auto loaded = rewriter.create< LLVM::LoadOp >(op.getLoc(),
                                                                  *trg_type,
                                                                  ops.getOperands()[0]);
                    rewriter.replaceOp(op, {loaded});
                    return mlir::success();
                }
                if (op.getKind() == hl::CastKind::IntegralCast)
                {
                    const auto &dl = this->type_converter().getDataLayoutAnalysis()
                                                           ->getAtOrAbove(op);
                    auto coerced = create_trunc_or_sext(
                            ops.getOperands()[0], *trg_type,
                            rewriter, op.getLoc(), dl);
                    rewriter.replaceOp(op, {coerced});
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
                auto target_ty = this->type_converter().convert_type_to_type(op.getType());
                auto new_op = rewriter.create< Trg >(op.getLoc(), *target_ty, ops.getOperands());
                rewriter.replaceOp(op, {new_op});
                return mlir::success();
            }
        };

        using one_to_one_conversions = util::type_list<
            one_to_one< hl::AddIOp, LLVM::AddOp >,
            one_to_one< hl::SubIOp, LLVM::SubOp >,
            one_to_one< hl::MulIOp, LLVM::MulOp >,

            one_to_one< hl::AddFOp, LLVM::FAddOp >,
            one_to_one< hl::SubFOp, LLVM::FSubOp >,
            one_to_one< hl::MulFOp, LLVM::FMulOp >,

            one_to_one< hl::DivSOp, LLVM::SDivOp >,
            one_to_one< hl::DivUOp, LLVM::UDivOp >,
            one_to_one< hl::DivFOp, LLVM::FDivOp >,

            one_to_one< hl::RemSOp, LLVM::SRemOp >,
            one_to_one< hl::RemUOp, LLVM::URemOp >,
            one_to_one< hl::RemFOp, LLVM::FRemOp >,

            one_to_one< hl::BinOrOp, LLVM::OrOp >,
            one_to_one< hl::BinAndOp, LLVM::AndOp >,
            one_to_one< hl::BinXorOp, LLVM::XOrOp >,

            one_to_one< hl::BinShlOp, LLVM::ShlOp >,
            one_to_one< hl::BinShlOp, LLVM::ShlOp >
        >;

        template< typename Src, typename Trg >
        struct assign_pattern : BasePattern< Src >
        {
            using Base = BasePattern< Src >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        Src op, typename Src::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto lhs = ops.getDst();
                auto rhs = ops.getSrc();

                // TODO(lukas): This should not happen?
                if (rhs.getType().template isa< hl::LValueType >())
                    return mlir::failure();

                auto load_lhs = rewriter.create< LLVM::LoadOp >(op.getLoc(), lhs);
                auto target_ty =
                    this->type_converter().convert_type_to_type(op.getSrc().getType());

                // Probably the easiest way to compose this (some template specialization would
                // require a lot of boilerplate).
                auto new_op = [&]()
                {
                    if constexpr (!std::is_same_v< Trg, void >)
                        return rewriter.create< Trg >(op.getLoc(), *target_ty, load_lhs, rhs);
                    else
                        return rhs;
                }();

                rewriter.create< LLVM::StoreOp >(op.getLoc(), new_op, lhs);

                // `hl.assign` returns value for cases like `int x = y = 5;`
                rewriter.replaceOp(op, {new_op});
                return mlir::success();
            }
        };

        using assign_conversions = util::type_list<
            assign_pattern< hl::AddIAssignOp, LLVM::AddOp >,
            assign_pattern< hl::SubIAssignOp, LLVM::SubOp >,
            assign_pattern< hl::AssignOp, void >
        >;


        struct call : BasePattern< hl::CallOp >
        {
            using Base = BasePattern< hl::CallOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        hl::CallOp op, typename hl::CallOp::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto module = op->getParentOfType< mlir::ModuleOp >();
                if (!module)
                    return mlir::failure();

                auto callee = module.lookupSymbol< mlir::LLVM::LLVMFuncOp >(op.getCallee());
                if (!callee)
                    return mlir::failure();

                auto rtys = this->type_converter().convert_types_to_types(
                        callee.getResultTypes());
                if (!rtys)
                    return mlir::failure();

                auto new_call = rewriter.create< mlir::LLVM::CallOp >(
                    op.getLoc(),
                    *rtys,
                    op.getCallee(),
                    ops.getOperands());
                rewriter.replaceOp(op, new_call.getResults());
                return mlir::success();
            }
        };

        bool is_lvalue(auto op)
        {
            return op && op.getType().template isa< hl::LValueType >();
        }

        struct prefix_tag {};
        struct postfix_tag {};

        template< typename Tag >
        constexpr static bool prefix_yield()
        {
            return std::is_same_v< Tag, prefix_tag >;
        }

        template< typename Tag >
        constexpr static bool postfix_yield()
        {
            return std::is_same_v< Tag, postfix_tag >;
        }

        template< typename Op, typename Trg, typename YieldAt >
        struct unary_in_place  : BasePattern< Op >
        {
            using Base = BasePattern< Op >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        Op op, typename Op::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto arg = ops.getArg();
                if (is_lvalue(arg))
                    return mlir::failure();

                auto value = rewriter.create< LLVM::LoadOp >(op.getLoc(), arg);
                auto one = this->iN(rewriter, op.getLoc(), value.getType(), 1);
                auto adjust = rewriter.create< Trg >(op.getLoc(), value, one);

                rewriter.create< LLVM::StoreOp >(op.getLoc(), adjust, arg);

                auto yielded = [&]() {
                    if constexpr (prefix_yield< YieldAt >())
                        return adjust;
                    else if constexpr (postfix_yield< YieldAt >())
                        return value;
                }();

                rewriter.replaceOp(op, {yielded});

                return mlir::success();
            }
        };

        using unary_in_place_conversions = util::type_list<
            unary_in_place< hl::PreIncOp,  LLVM::AddOp, prefix_tag  >,
            unary_in_place< hl::PostIncOp, LLVM::AddOp, postfix_tag >,

            unary_in_place< hl::PreDecOp,  LLVM::SubOp, prefix_tag  >,
            unary_in_place< hl::PostDecOp, LLVM::SubOp, postfix_tag >
        >;

        struct cmp : BasePattern< hl::CmpOp >
        {

            using Base = BasePattern< hl::CmpOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        hl::CmpOp op, typename hl::CmpOp::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto predicate = convert_predicate(op.getPredicate());
                if (!predicate)
                    return mlir::failure();

                auto new_cmp = rewriter.create< mlir::LLVM::ICmpOp >(
                    op.getLoc(), *predicate, ops.getLhs(), ops.getRhs());
                rewriter.replaceOp(op, { new_cmp });
                return mlir::success();
            }

            auto convert_predicate(auto hl_predicate) const
                -> std::optional< mlir::LLVM::ICmpPredicate >
            {
                // TODO(lukas): Use map later, this is just a proof of concept.
                switch (hl_predicate)
                {
                    case hl::Predicate::eq : return { mlir::LLVM::ICmpPredicate::eq };
                    case hl::Predicate::ne : return { mlir::LLVM::ICmpPredicate::ne };
                    case hl::Predicate::slt : return { mlir::LLVM::ICmpPredicate::slt };
                    case hl::Predicate::sle : return { mlir::LLVM::ICmpPredicate::sle };
                    case hl::Predicate::sgt : return { mlir::LLVM::ICmpPredicate::sgt };
                    case hl::Predicate::sge : return { mlir::LLVM::ICmpPredicate::sge };
                    case hl::Predicate::ult : return { mlir::LLVM::ICmpPredicate::ult };
                    case hl::Predicate::ule : return { mlir::LLVM::ICmpPredicate::ule };
                    case hl::Predicate::ugt : return { mlir::LLVM::ICmpPredicate::ugt };
                    case hl::Predicate::uge : return { mlir::LLVM::ICmpPredicate::uge };
                }

                VAST_UNREACHABLE("unsupported predicate");
            }
        };

        struct deref : BasePattern< hl::Deref >
        {
            using Base = BasePattern< hl::Deref >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                        hl::Deref op, typename hl::Deref::Adaptor ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto trg_type = tc.convert_type_to_type(op.getType());
                if (!trg_type)
                    return mlir::failure();

                auto loaded = rewriter.create< mlir::LLVM::LoadOp >(
                        op.getLoc(), *trg_type, ops.getAddr());
                rewriter.replaceOp(op, { loaded });

                return mlir::success();
            }
        };

        using base_op_conversions = util::type_list<
            func_op,
            constant_int,
            ret,
            implicit_cast,
            call,
            cmp,
            deref
        >;

        // Drop types of operations that will be processed by pass for core(lazy) operations.
        template< typename LazyOp >
        struct lazy_op_type : BasePattern< LazyOp >
        {
            using Base = BasePattern< LazyOp >;
            using Base::Base;

            mlir::LogicalResult matchAndRewrite(
                LazyOp op, typename LazyOp::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto lower_res_type = [&]()
                {
                    auto result = op.getResult();
                    result.setType(this->type_converter().convertType(result.getType()));
                };

                rewriter.updateRootInPlace(op, lower_res_type);

                return mlir::success();
            }
        };

        using lazy_op_type_conversions = util::type_list<
            lazy_op_type< core::LazyOp >,
            lazy_op_type< core::BinLAndOp >,
            lazy_op_type< core::BinLOrOp >,
            lazy_op_type< hl::ValueYieldOp >
        >;
    } // namespace pattern


    template< typename Op >
    bool has_llvm_only_types(Op op)
    {
        return util::for_each_subtype(op.getResultTypes(), mlir::LLVM::isCompatibleType);
    }

    template < typename Op >
    bool has_llvm_return_type(Op op)
    {
        return mlir::LLVM::isCompatibleType(op.getResult().getType());
    }

    struct IRsToLLVMPass : ModuleLLVMConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >
    {
        using base = ModuleLLVMConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >;

        static conversion_target create_conversion_target(MContext &context) {
            conversion_target target(context);

            target.addIllegalDialect< hl::HighLevelDialect >();
            target.addIllegalDialect< ll::LowLevelDialect >();
            target.addLegalDialect< core::CoreDialect>();
            target.addLegalOp< hl::TypeDefOp >();
            target.addLegalOp< hl::ValueYieldOp>();

            target.addDynamicallyLegalOp< core::LazyOp >(has_llvm_return_type< core::LazyOp >);
            target.addDynamicallyLegalOp< core::BinLAndOp >(has_llvm_return_type< core::BinLAndOp>);
            target.addDynamicallyLegalOp< core::BinLOrOp >(has_llvm_return_type< core::BinLOrOp>);
            target.addDynamicallyLegalOp< hl::ValueYieldOp >(has_llvm_return_type< hl::ValueYieldOp >);

            target.addDynamicallyLegalOp< hl::InitListExpr >(
                    has_llvm_only_types< hl::InitListExpr>);

            target.addIllegalOp< mlir::func::FuncOp >();
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });

            return target;
        }

        static void populate_conversions(rewrite_pattern_set &patterns, type_converter &converter) {
            base::populate_conversions_base<
                pattern::one_to_one_conversions,
                pattern::inline_region_from_op_conversions,
                pattern::assign_conversions,
                pattern::init_conversions,
                pattern::base_op_conversions,
                pattern::ignore_patterns,
                pattern::lazy_op_type_conversions
            >(patterns, converter);
        }
    };
} // namespace vast


std::unique_ptr< mlir::Pass > vast::createIRsToLLVMPass()
{
    return std::make_unique< vast::IRsToLLVMPass >();
}
