// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{
    namespace
    {
        [[maybe_unused]] std::size_t size(mlir::Region &region)
        {
            return std::distance(region.begin(), region.end());
        }
        [[maybe_unused]] std::size_t size(mlir::Block &block)
        {
            return std::distance(block.begin(), block.end());
        }
    }

    template< typename T >
    auto inline_cond_region(auto src, auto &rewriter)
    {
        auto up = src->getParentRegion();
        VAST_ASSERT(up != nullptr);

        auto &copy = src.condRegion();
        VAST_ASSERT(size(copy) == 1);

        rewriter.cloneRegionBefore(copy, *up, up->end());
        rewriter.mergeBlockBefore(&up->back(), src);

        return mlir::dyn_cast< T >(*std::prev(mlir::Block::iterator(src)));
    }

    namespace pattern
    {
        auto coerce_condition(auto op, mlir::ConversionPatternRewriter &rewriter)
        -> std::optional< mlir::Value >
        {
            auto int_type = op.getType().template dyn_cast< mlir::IntegerType >();
            if (!int_type)
                return {};

            auto i1 =  mlir::IntegerType::get(op.getContext(), 1u);
            if (int_type == i1)
                return { op };

            auto coerced = rewriter.create< hl::ImplicitCastOp >(
                op.getLoc(), i1, op, hl::CastKind::IntegralCast);
            return { coerced };
        }

        template< typename T >
        struct DoConversion {};

        template< typename O >
        struct State
        {
            O op;
            typename O::Adaptor operands;
            mlir::ConversionPatternRewriter &rewriter;

            State(O op_, typename O::Adaptor operands_,
                  mlir::ConversionPatternRewriter &rewriter_)
                : op(op_), operands(operands_), rewriter(rewriter_)
            {}
        };

        template<>
        struct DoConversion< hl::IfOp > : State < hl::IfOp >
        {
            using State< hl::IfOp >::State;

            mlir::Region &steal_region(mlir::Region &from, mlir::Region &into)
            {
                rewriter.inlineRegionBefore(from, into, into.begin());
                return into;
            }

            mlir::Region &remove_back(mlir::Region &region)
            {
                VAST_ASSERT(size(region) >= 1);
                rewriter.eraseBlock(&region.back());
                return region;
            }

            std::optional< mlir::Block * > get_singleton_block(mlir::Region &region)
            {
                if (size(region) != 1)
                    return {};
                return { &region.front() };

            }

            bool has_terminator(mlir::Block &block)
            {
                if (size(block) == 0)
                    return false;

                auto &last = block.back();
                return last.hasTrait< mlir::OpTrait::IsTerminator >();
            }

            std::optional< mlir::Operation * > get_terminator(mlir::Block &block)
            {
                if (has_terminator(block))
                    return { block.getTerminator() };
                return {};
            }

            template< typename O, typename GetVals >
            auto new_terminator(mlir::Block &block, GetVals &&get_vals)
            -> std::optional< mlir::Operation * >
            {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToEnd(&block);

                auto maybe_terminator = get_terminator(block);
                rewriter.template create< O >(op.getLoc(), get_vals(maybe_terminator));
                return maybe_terminator;
            }

            template< typename O, typename GetVals >
            void replace_terminator(mlir::Block &block, GetVals &&get_vals)
            {
                auto old = new_terminator< O >(block, std::forward< GetVals >(get_vals));
                if (old)
                    rewriter.eraseOp(*old);
            }

            auto no_vals()
            {
                // `std::optional< mlir::Operation * > -> std::vector< mlir::Value * >
                return [](auto) { return std::vector< mlir::Value >{}; };
            }

            auto erase_op()
            {
                return [&](auto op) { return rewriter.eraseOp(op); };
            }

            mlir::LogicalResult make_if_block(mlir::Region &from, mlir::Region &to)
            {
                auto block = get_singleton_block(remove_back(steal_region(from, to)));
                if (!block)
                    return mlir::failure();
                replace_terminator< mlir::scf::YieldOp >(**block, no_vals());
                return mlir::success();
            }

            template< typename H, typename ... Args >
            bool failed(H &&h, Args && ... args)
            {
                if (mlir::failed(std::forward< H >(h)))
                    return true;

                if constexpr (sizeof ... (Args) != 0)
                    return failed< Args ... >(std::forward< Args >(args) ...);
                return false;
            }

            mlir::LogicalResult convert()
            {
                auto yield = inline_cond_region< hl::CondYieldOp >(op, rewriter);
                rewriter.setInsertionPointAfter(yield);

                auto coerced = coerce_condition(yield.getOperand(), rewriter);
                if (!coerced)
                    return mlir::failure();

                mlir::scf::IfOp scf_if_op = rewriter.create< mlir::scf::IfOp >(
                        op.getLoc(), std::vector< mlir::Type >{}, *coerced,
                        op.hasElse());
                auto then_result = make_if_block(op.thenRegion(), scf_if_op.getThenRegion());
                auto else_result = [&]()
                {
                    if (op.hasElse())
                        return make_if_block(op.elseRegion(), scf_if_op.getElseRegion());
                    return mlir::success();
                }();
                if (failed(then_result, else_result))
                    return mlir::failure();

                rewriter.eraseOp(yield);
                rewriter.eraseOp(op);

                return mlir::success();
            }
        };

        template< typename T >
        struct BasePattern : mlir::ConvertOpToLLVMPattern< T >
        {
            using Base = mlir::ConvertOpToLLVMPattern< hl::IfOp >;
            using Base::Base;
            using operation_t = T;

            mlir::LogicalResult matchAndRewrite(
                    T op, typename T::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                return DoConversion< T >(op, ops, rewriter).convert();
            }
        };

        using l_ifop = BasePattern< hl::IfOp >;

        struct l_while : mlir::ConvertOpToLLVMPattern< hl::WhileOp >
        {
            using Base = mlir::ConvertOpToLLVMPattern< hl::WhileOp >;
            using Base::Base;


            mlir::Block &do_inline(mlir::Region &src, mlir::Region &dst,
                                   mlir::ConversionPatternRewriter &rewriter) const
            {
                rewriter.createBlock(&dst);
                rewriter.cloneRegionBefore(src, &dst.back());
                rewriter.eraseBlock(&dst.back());

                return dst.back();
            }

            template< typename T >
            std::optional< T > fetch_terminator(mlir::Block &block) const
            {
                auto out = mlir::dyn_cast< T >(block.getTerminator());
                if (!out)
                    return {};
                return out;
            }

            mlir::LogicalResult before_region(mlir::Block &dst,
                                              mlir::ConversionPatternRewriter &rewriter) const
            {
                auto cond_yield = fetch_terminator< hl::CondYieldOp >(dst);
                if (!cond_yield)
                    return mlir::failure();

                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointAfter(*cond_yield);
                rewriter.create< mlir::scf::ConditionOp >(
                        cond_yield->getLoc(),
                        cond_yield->result(),
                        dst.getParent()->front().getArguments());

                rewriter.eraseOp(*cond_yield);
                return mlir::success();
            }

            mlir::LogicalResult after_region(mlir::Block &block, mlir::Block &before,
                                             mlir::Location loc,
                                             mlir::ConversionPatternRewriter &rewriter) const
            {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToEnd(&block);
                rewriter.create< mlir::scf::YieldOp >(loc, before.getArguments());

                return mlir::success();
            }

            mlir::LogicalResult matchAndRewrite(
                    hl::WhileOp op, hl::WhileOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto scf_while_op = rewriter.create< mlir::scf::WhileOp >(
                        op.getLoc(),
                        std::vector< mlir::Type >{},
                        std::vector< mlir::Value >{});
                auto &before = do_inline(op.condRegion(), scf_while_op.getBefore(), rewriter);
                auto &after = do_inline(op.bodyRegion(), scf_while_op.getAfter(), rewriter);

                if (mlir::failed(before_region(before, rewriter)) ||
                    mlir::failed(after_region(after, before, op.getLoc(), rewriter)))
                {
                    return mlir::failure();
                }

                rewriter.eraseOp(op);
                return mlir::success();
            }
        };

    } // namespace pattern


    struct LowerHighLevelControlFlowPass : LowerHighLevelControlFlowBase< LowerHighLevelControlFlowPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelControlFlowPass::runOnOperation()
    {
        auto op = this->getOperation();
        auto &mctx = this->getContext();

        mlir::ConversionTarget trg(mctx);
        trg.addIllegalOp< hl::IfOp >();
        trg.addIllegalOp< hl::WhileOp >();
        trg.addIllegalOp< hl::CondYieldOp >();
        trg.addLegalOp< mlir::scf::IfOp >();
        trg.addLegalOp< mlir::scf::YieldOp >();
        trg.addLegalOp< mlir::scf::ConditionOp >();
        trg.markUnknownOpDynamicallyLegal([](auto) { return true; });

        mlir::RewritePatternSet patterns(&mctx);

        mlir::LowerToLLVMOptions llvm_opts{ &mctx };
        const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();

        auto tc = mlir::LLVMTypeConverter(&mctx, llvm_opts, &dl_analysis);
        patterns.add< pattern::l_ifop >(tc);
        patterns.add< pattern::l_while >(tc);
        if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
            return signalPassFailure();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelControlFlowPass()
{
  return std::make_unique< LowerHighLevelControlFlowPass >();
}
