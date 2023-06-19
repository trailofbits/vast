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
#include "vast/Dialect/Core/TypeTraits.hpp"

#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/Terminator.hpp"
#include "vast/Util/TypeList.hpp"
#include "vast/Util/Common.hpp"

#include "vast/Conversion/Common/Passes.hpp"

#include "Common.hpp"
#include "LLCFToLLVM.hpp"

namespace vast::conv::irstollvm
{
    using ignore_patterns = util::type_list<
        ignore_pattern< hl::DeclRefOp >
    >;

    template< typename Op >
    struct inline_region_from_op : base_pattern< Op >
    {
        using base = base_pattern< Op >;
        using base::base;

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


    struct uninit_var : base_pattern< ll::UninitializedVar >
    {
        using op_t = ll::UninitializedVar;
        using base = base_pattern< op_t >;
        using base::base;

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

    struct initialize_var : base_pattern< ll::InitializeVar >
    {
        using op_t = ll::InitializeVar;
        using base = base_pattern< op_t >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                op_t op, typename op_t::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto index_type = tc.convert_type_to_type(rewriter.getIndexType());
            VAST_PATTERN_CHECK(index_type, "Was not able to convert index type");

            for (auto element : ops.getElements())
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
                                op.getLoc(), e_type, ops.getVar(), index.getResult());

                        rewriter.create< LLVM::StoreOp >(op.getLoc(), expr_elem, gep);
                    }
                    rewriter.eraseOp(init_list_expr);
                    break;
                }

                rewriter.create< LLVM::StoreOp >(op.getLoc(), element, ops.getVar());
            }

            // While op is a value, there is no reason not to use the previous alloca,
            // since we just initialized it.
            rewriter.replaceOp(op, ops.getVar());

            return mlir::success();
        }
    };

    struct init_list_expr : base_pattern< hl::InitListExpr >
    {
        using op_t = hl::InitListExpr;
        using base = base_pattern< op_t >;
        using base::base;

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

    template< typename Op >
    struct func_op : base_pattern< Op >
    {
        using op_t = Op;
        using base = base_pattern< op_t >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                op_t func_op, typename op_t::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto &tc = this->type_converter();

            auto maybe_target_type = tc.convert_fn_t(func_op.getFunctionType());
            // TODO(irs-to-llvm): Handle varargs.
            auto maybe_signature =
                tc.get_conversion_signature(func_op, /* variadic */ true);

            // Type converter failed.
            if (!maybe_target_type || !*maybe_target_type || !maybe_signature)
            {
                VAST_PATTERN_FAIL("Failed to convert function type.");
            }

            auto target_type = *maybe_target_type;
            auto signature = *maybe_signature;

            // TODO(irs-to-llvm): Currently it is unclear what to do with the
            //                    arg/res attributes as it looks like we may not
            //                    want to lower them all.


            // TODO(lukas): Linkage?
            auto linkage = LLVM::Linkage::External;
            auto new_func = rewriter.create< LLVM::LLVMFuncOp >(
                    func_op.getLoc(), func_op.getName(), target_type,
                    linkage, false, LLVM::CConv::C);
            rewriter.inlineRegionBefore(func_op.getBody(),
                                        new_func.getBody(), new_func.end());
            util::convert_region_types(func_op, new_func, signature);

            if (mlir::failed(args_to_allocas(new_func, rewriter)))
            {
                VAST_PATTERN_FAIL("Failed to convert func arguments");
            }
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
                return mlir::success();

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
                    this->type_converter().convertType(rewriter.getIndexType()),
                    rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

            auto alloca_op = rewriter.create< LLVM::AllocaOp >(
                    arg.getLoc(), ptr_type, count, 0);

            arg.replaceAllUsesWith(alloca_op);
            rewriter.create< mlir::LLVM::StoreOp >(arg.getLoc(), arg, alloca_op);

            return mlir::success();
        }

        static void legalize(conversion_target &target)
        {
            target.addIllegalOp< op_t >();
        }
    };

    struct constant_int : base_pattern< hl::ConstantOp >
    {
        using base = base_pattern< hl::ConstantOp >;
        using base::base;

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

    template< typename return_like_op >
    struct ret : base_pattern< return_like_op >
    {
        using op_t = return_like_op;
        using base = base_pattern< op_t >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                op_t ret_op, typename op_t::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            rewriter.create< LLVM::ReturnOp >(ret_op.getLoc(), ops.getOperands());
            rewriter.eraseOp(ret_op);
            return mlir::success();
        }

        static void legalize(conversion_target &target)
        {
            target.addIllegalOp< return_like_op >();
        }
    };

    using return_conversions = util::type_list<
          ret< hl::ReturnOp >
        , ret< ll::ReturnOp >
        , ret< mlir::func::ReturnOp >
    >;

    struct implicit_cast : base_pattern< hl::ImplicitCastOp >
    {
        using base = base_pattern< hl::ImplicitCastOp >;
        using base::base;

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
                return logical_result::success();
            }
            if (op.getKind() == hl::CastKind::IntegralCast)
            {
                const auto &dl = this->type_converter().getDataLayoutAnalysis()
                                                       ->getAtOrAbove(op);
                auto coerced = create_trunc_or_sext(
                        ops.getOperands()[0], *trg_type,
                        rewriter, op.getLoc(), dl);
                rewriter.replaceOp(op, {coerced});
                return logical_result::success();
            }
            if (op.getKind() == hl::CastKind::IntegralToFloating)
            {
                if (op.getOperand().getType().isUnsignedInteger())
                {
                    rewriter.replaceOpWithNewOp< LLVM::UIToFPOp >(op, *trg_type, ops.getValue());
                    return logical_result::success();
                }
                if (op.getOperand().getType().isSignedInteger())
                {
                    rewriter.replaceOpWithNewOp< LLVM::SIToFPOp >(op, *trg_type, ops.getValue());
                    return logical_result::success();
                }
            }
            return logical_result::failure();
        }
    };

    struct cstyle_cast : base_pattern< hl::CStyleCastOp >
    {
        using op_t = hl::CStyleCastOp;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
                    op_t op, typename op_t::Adaptor ops,
                    conversion_rewriter &rewriter) const override
        {
            // TODO: According to what clang does, this will need more handling
            //       based on different value categories. For now, just lower types.
            auto to_void = [&]
            {
                rewriter.replaceOp(op, ops.getOperands());
                return mlir::success();
            };

            switch (op.getKind())
            {
                case hl::CastKind::ToVoid:
                    return to_void();
                default:
                    return mlir::failure();
            }
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
    struct assign_pattern : base_pattern< Src >
    {
        using base = base_pattern< Src >;
        using base::base;

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


    struct call : base_pattern< hl::CallOp >
    {
        using base = base_pattern< hl::CallOp >;
        using base::base;

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

            auto mk_call = [&](auto ... args)
            {
                return rewriter.create< mlir::LLVM::CallOp >(op.getLoc(), args ...);
            };

            if (rtys->empty() || rtys->front().isa< mlir::LLVM::LLVMVoidType >())
            {
                // We cannot pass in void type as some internal check inside `mlir::LLVM`
                // dialect will fire - it would create a value of void type, which is
                // not allowed.
                mk_call(std::vector< mlir::Type >{}, op.getCallee(), ops.getOperands());
                rewriter.eraseOp(op);
            } else {
                auto call = mk_call(*rtys, op.getCallee(), ops.getOperands());
                rewriter.replaceOp(op, call.getResults());
            }

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
    struct unary_in_place  : base_pattern< Op >
    {
        using base = base_pattern< Op >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                    Op op, typename Op::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto arg = ops.getArg();
            if (is_lvalue(arg))
                return mlir::failure();

            auto value = rewriter.create< LLVM::LoadOp >(op.getLoc(), arg);
            auto one = this->constant(rewriter, op.getLoc(), value.getType(), 1);
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

    struct logical_not : base_pattern< hl::LNotOp >
    {
        using base = base_pattern< hl::LNotOp >;
        using base::base;
        using adaptor_t = typename hl::LNotOp::Adaptor;
        using integer_t = mlir::IntegerType;

        Operation * create_cmp(conversion_rewriter &rewriter, hl::LNotOp &op, Value lhs, Value rhs) const {
            auto lhs_type = lhs.getType();

            if (llvm::isa< mlir::FloatType >(lhs_type)) {
                auto i1_type = integer_t::get(op.getContext(), 1, integer_t::Signless);

                return rewriter.create< LLVM::FCmpOp >(
                            op.getLoc(), i1_type,
                            LLVM::FCmpPredicate::une,
                            lhs, rhs
                       );
            }

            if (llvm::isa< mlir::IntegerType >(lhs_type)) {
                return rewriter.create< LLVM::ICmpOp >(
                    op.getLoc(),
                    LLVM::ICmpPredicate::ne,
                    lhs, rhs
                );
            }
            VAST_UNREACHABLE("Unknwon cmp type.");
        }

        logical_result matchAndRewrite(hl::LNotOp op, adaptor_t adaptor, conversion_rewriter &rewriter) const override {
            auto zero = this->constant(rewriter, op.getLoc(), adaptor.getArg().getType(), 0);

            auto cmp = create_cmp(rewriter, op, adaptor.getArg(), zero);

            auto true_i1 = this->iN(rewriter, op.getLoc(), cmp->getResult(0).getType(), 1);
            auto xor_val = rewriter.create< LLVM::XOrOp >(op.getLoc(), cmp->getResult(0), true_i1);

            auto res_type = this->type_converter().convert_type_to_type(op.getResult().getType());
            if (!res_type)
                return logical_result::failure();
            rewriter.replaceOpWithNewOp< LLVM::ZExtOp >(op, *res_type, xor_val);
            return logical_result::success();
        }

    };

    using unary_in_place_conversions = util::type_list<
        unary_in_place< hl::PreIncOp,  LLVM::AddOp, prefix_tag  >,
        unary_in_place< hl::PostIncOp, LLVM::AddOp, postfix_tag >,

        unary_in_place< hl::PreDecOp,  LLVM::SubOp, prefix_tag  >,
        unary_in_place< hl::PostDecOp, LLVM::SubOp, postfix_tag >,
        logical_not
    >;

    struct cmp : base_pattern< hl::CmpOp >
    {

        using base = base_pattern< hl::CmpOp >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                    hl::CmpOp op, typename hl::CmpOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto predicate = convert_predicate(op.getPredicate());
            if (!predicate)
                return mlir::failure();

            auto new_cmp = rewriter.create< mlir::LLVM::ICmpOp >(
                op.getLoc(), *predicate, ops.getLhs(), ops.getRhs());

            auto trg_type = tc.convert_type_to_type(op.getType());
            if (!trg_type)
                return mlir::failure();

            auto coerced = create_trunc_or_sext(new_cmp, *trg_type,
                                                rewriter, op.getLoc(), dl(op));

            rewriter.replaceOp(op, { coerced });
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

    struct deref : base_pattern< hl::Deref >
    {
        using base = base_pattern< hl::Deref >;
        using base::base;

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

    template< typename op_t, typename yield_op_t >
    struct propagate_yield : base_pattern< op_t >
    {
        using base = base_pattern< op_t >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
                    op_t op, typename op_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto body = op.getBody();
            if (!body)
                return mlir::success();

            auto yield = terminator_t< yield_op_t >::get(*body);
            VAST_PATTERN_CHECK(yield, "Expected yield in: {0}", op);

            rewriter.mergeBlockBefore(body, op);
            rewriter.replaceOp(op, yield.op().getResult());
            rewriter.eraseOp(yield.op());
            return mlir::success();
        }
    };

    using base_op_conversions = util::type_list<
        func_op< mlir::func::FuncOp >,
        func_op< hl::FuncOp >,
        constant_int,
        implicit_cast,
        cstyle_cast,
        call,
        cmp,
        deref,
        propagate_yield< hl::ExprOp, hl::ValueYieldOp >
    >;

    // Drop types of operations that will be processed by pass for core(lazy) operations.
    template< typename LazyOp >
    struct lazy_op_type : base_pattern< LazyOp >
    {
        using base = base_pattern< LazyOp >;
        using base::base;

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

    template< typename yield_like >
    struct fixup_yield_types : base_pattern< yield_like >
    {
        using op_t = yield_like;
        using base = base_pattern< yield_like >;
        using base::base;

        mlir::LogicalResult matchAndRewrite(
            op_t op, typename op_t::Adaptor ops,
            mlir::ConversionPatternRewriter &rewriter) const override
        {
            if (ops.getResult().getType().template isa< mlir::LLVM::LLVMVoidType >())
            {
                rewriter.eraseOp(op);
                return mlir::success();
            }

            // Proceed with "normal path"
            // TODO: This will probably need rework. If not, merge with implementation
            //       in `lazy_op_type`.
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
        fixup_yield_types< hl::ValueYieldOp >
    >;

    struct IRsToLLVMPass : ModuleLLVMConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >
    {
        using base = ModuleLLVMConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >;
        using config_t = typename base::config_t;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);

            target.addIllegalDialect< hl::HighLevelDialect >();
            target.addIllegalDialect< ll::LowLevelDialect >();
            target.addLegalDialect< core::CoreDialect >();

            target.addLegalOp< hl::TypeDefOp >();

            auto illegal_with_llvm_ret_type = [&]< typename T >( T && )
            {
                target.addDynamicallyLegalOp< T >( has_llvm_return_type< T > );
            };

            illegal_with_llvm_ret_type( core::LazyOp{} );
            illegal_with_llvm_ret_type( core::BinLAndOp{} );
            illegal_with_llvm_ret_type( core::BinLOrOp{} );
            illegal_with_llvm_ret_type( hl::ValueYieldOp{} );


            target.addDynamicallyLegalOp< hl::InitListExpr >(
                has_llvm_only_types< hl::InitListExpr >
            );

            target.addIllegalOp< mlir::func::FuncOp >();
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });

            return target;
        }

        static void populate_conversions(config_t &config) {
            base::populate_conversions_base<
                one_to_one_conversions,
                inline_region_from_op_conversions,
                return_conversions,
                assign_conversions,
                unary_in_place_conversions,
                init_conversions,
                base_op_conversions,
                ignore_patterns,
                lazy_op_type_conversions,
                ll_cf::conversions
            >(config);
        }

        static void set_llvm_opts(mlir::LowerToLLVMOptions &llvm_options) {
            llvm_options.useBarePtrCallConv = true;
        }

    };
} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createIRsToLLVMPass()
{
    return std::make_unique< vast::conv::irstollvm::IRsToLLVMPass >();
}
