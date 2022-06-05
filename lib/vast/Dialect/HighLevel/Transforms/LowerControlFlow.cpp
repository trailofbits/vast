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

        struct optional_terminator_t : std::optional< mlir::Operation * >
        {
            template< typename T >
            T cast() const
            {
                if (!has_value())
                    return {};
                return mlir::dyn_cast< T >(**this);
            }
        };

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

            bool has_terminator(mlir::Block &block) const
            {
                if (size(block) == 0)
                    return false;

                auto &last = block.back();
                return last.hasTrait< mlir::OpTrait::IsTerminator >();
            }

            optional_terminator_t get_terminator(mlir::Block &block) const
            {
                if (has_terminator(block))
                    return { block.getTerminator() };
                return {};
            }

            std::optional< mlir::Block * > get_singleton_block(mlir::Region &region)
            {
                if (size(region) != 1)
                    return {};
                return { &region.front() };

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

            mlir::LogicalResult after_region(mlir::Block &block, mlir::Block &before,
                                             mlir::Location loc) const
            {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToEnd(&block);
                rewriter.create< mlir::scf::YieldOp >(loc, before.getArguments());

                return mlir::success();
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
                    mlir::failed(after_region(after, before, op.getLoc())))
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


    struct LowerHighLevelControlFlowPass : LowerHighLevelControlFlowBase< LowerHighLevelControlFlowPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelControlFlowPass::runOnOperation()
    {
        auto op = this->getOperation();
        auto &mctx = this->getContext();

        mlir::ConversionTarget trg(mctx);
        trg.addLegalDialect< mlir::scf::SCFDialect >();

        trg.addIllegalOp< hl::IfOp >();
        trg.addIllegalOp< hl::WhileOp >();
        trg.addIllegalOp< hl::CondYieldOp >();

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
