// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Terminator.hpp"

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

        template< typename Op >
        struct State
        {
            Op op;
            typename Op::Adaptor operands;
            mlir::ConversionPatternRewriter &rewriter;

            State(Op op, typename Op::Adaptor operands,
                  mlir::ConversionPatternRewriter &rewriter)
                : op(op), operands(operands), rewriter(rewriter)
            {}

            std::optional< mlir::Block * > get_singleton_block(mlir::Region &region)
            {
                if (size(region) != 1)
                    return {};
                return { &region.front() };

            }

            mlir::LogicalResult wrap_result(bool c)
            {
                return (c) ? mlir::success() : mlir::failure();
            }

            mlir::LogicalResult emit_scf_yield(mlir::Block &block,
                                               const std::vector< mlir::Value > &args)
            {
                if (auto terminator = get_terminator(block))
                {
                    return wrap_result(terminator.cast< mlir::scf::YieldOp >());
                }

                mlir::ConversionPatternRewriter::InsertionGuard guard{ rewriter };
                rewriter.setInsertionPointToEnd(&block);

                rewriter.create< mlir::scf::YieldOp >(op.getLoc(),
                                                      args);
                return mlir::success();
            }

            mlir::LogicalResult wrap_hl_return(mlir::Block &block,
                                               const std::vector< mlir::Value > &args = {})
            {
                auto hl_ret = get_terminator(block).cast< hl::ReturnOp >();
                if (!hl_ret)
                    return emit_scf_yield(block, args);

                auto scope = [&]()
                {
                    mlir::ConversionPatternRewriter::InsertionGuard guard{ rewriter };
                    rewriter.setInsertionPointToEnd(&block);
                    return rewriter.create< hl::ScopeOp >(op.getLoc());
                }();
                auto wrap_block = [&]() -> mlir::Block *
                {
                    VAST_ASSERT(size(scope.body()) <= 1);
                    if (size(scope.body()) == 0)
                        rewriter.createBlock(&scope.body());
                    return &scope.body().front();
                }();

                // NOTE(lukas): Did not find api to move operations around.
                {
                    mlir::ConversionPatternRewriter::InsertionGuard guard{ rewriter };
                    rewriter.setInsertionPointToEnd(wrap_block);
                    rewriter.clone(*hl_ret.getOperation());
                    rewriter.eraseOp(hl_ret);
                }

                return emit_scf_yield(block, args);
            }
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

            mlir::LogicalResult make_if_block(mlir::Region &from, mlir::Region &to)
            {
                auto block = get_singleton_block(remove_back(steal_region(from, to)));
                if (!block)
                    return mlir::failure();
                return wrap_hl_return(**block);
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

        template<>
        struct DoConversion< hl::WhileOp > : State< hl::WhileOp >
        {
            using State< hl::WhileOp >::State;

            mlir::Block &do_inline(mlir::Region &src, mlir::Region &dst) const
            {
                rewriter.createBlock(&dst);
                rewriter.cloneRegionBefore(src, &dst.back());
                rewriter.eraseBlock(&dst.back());

                return dst.back();
            }

            mlir::LogicalResult before_region(mlir::Block &dst) const
            {
                auto cond_yield = get_terminator(dst).cast< hl::CondYieldOp >();
                if (!cond_yield)
                    return mlir::failure();

                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointAfter(cond_yield);
                auto coerced_condition = coerce_condition(cond_yield.result(), rewriter);
                if (!coerced_condition)
                    return mlir::failure();

                rewriter.create< mlir::scf::ConditionOp >(
                        cond_yield.getLoc(),
                        *coerced_condition,
                        dst.getParent()->front().getArguments());

                rewriter.eraseOp(cond_yield);
                return mlir::success();
            }

            mlir::LogicalResult after_region(mlir::Block &block, mlir::Block &before)
            {
                std::vector< mlir::Value > vals;
                for (auto x : before.getArguments())
                    vals.push_back(x);
                return wrap_hl_return(block, vals);
            }

            // NOTE(lukas): Could be `const` but since `op.getLoc()` is not it does not compile.
            mlir::LogicalResult convert()
            {
                auto scf_while_op = rewriter.create< mlir::scf::WhileOp >(
                        op.getLoc(),
                        std::vector< mlir::Type >{},
                        std::vector< mlir::Value >{});
                auto &before = do_inline(op.condRegion(), scf_while_op.getBefore());
                auto &after = do_inline(op.bodyRegion(), scf_while_op.getAfter());

                if (mlir::failed(before_region(before)) ||
                    mlir::failed(after_region(after, before)))
                {
                    return mlir::failure();
                }

                rewriter.eraseOp(op);
                return mlir::success();
            }
        };

        // TODO(lukas): Unfortunately `scf.for` may be too narrow to be always be used
        //              as valid target for `hl.for` lowering - for example `step` is always
        //              required, whereas `hl.for` support arbitrary computation in its `incr`
        //              block.
        //              We may be able to lower some `hl.for` into `scf.for` but I think
        //              we will need to provide a generic pattern into `std` for the rest.
        template<>
        struct DoConversion< hl::ForOp > : State< hl::ForOp >
        {
            using State< hl::ForOp >::State;

            mlir::LogicalResult convert()
            {
                return mlir::failure();
            }
        };

        template< typename T >
        struct BasePattern : mlir::ConvertOpToLLVMPattern< T >
        {
            using Base = mlir::ConvertOpToLLVMPattern< T >;
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
        using l_while = BasePattern< hl::WhileOp >;
        using l_for = BasePattern< hl::ForOp >;

    } // namespace pattern


    struct HLToSCFPass : HLToSCFBase< HLToSCFPass >
    {
        void runOnOperation() override;
    };

    void HLToSCFPass::runOnOperation()
    {
        auto op = this->getOperation();
        auto &mctx = this->getContext();

        mlir::ConversionTarget trg(mctx);
        trg.addLegalDialect< mlir::scf::SCFDialect >();

        trg.addIllegalOp< hl::IfOp,
                          hl::WhileOp,
                          hl::CondYieldOp >();

        trg.markUnknownOpDynamicallyLegal([](auto) { return true; });

        mlir::RewritePatternSet patterns(&mctx);

        mlir::LowerToLLVMOptions llvm_opts{ &mctx };
        const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();

        auto tc = mlir::LLVMTypeConverter(&mctx, llvm_opts, &dl_analysis);
        patterns.add< pattern::l_ifop,
                      pattern::l_while >(tc);
        if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
            return signalPassFailure();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createHLToSCFPass()
{
  return std::make_unique< HLToSCFPass >();
}
