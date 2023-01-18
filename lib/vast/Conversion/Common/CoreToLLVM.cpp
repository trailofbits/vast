// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/BlockAndValueMapping.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"
#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Util/TypeList.hpp"
#include "vast/Util/DialectConversion.hpp"

namespace vast
{
    namespace pattern
    {
        namespace LLVM = mlir::LLVM;

        using conversion_rewriter = mlir::ConversionPatternRewriter;

        template< typename LOp, bool short_on_true >
        struct lazy_bin_logical : operation_conversion_pattern< LOp >
        {
            using base = operation_conversion_pattern< LOp >;
            using base::base;
            using adaptor_t = typename LOp::Adaptor;

            auto insert_lazy_to_block(mlir::Operation* lazy_op, mlir::Block* target,
                conversion_rewriter &rewriter) const
            {
                auto &lazy_region = lazy_op->getRegion(0);
                auto &lazy_block = lazy_region.front();

                auto &yield = lazy_block.back();
                auto res = yield.getOperand(0);
                rewriter.eraseOp(&yield);

                rewriter.inlineRegionBefore(lazy_region, *target->getParent(), ++(target->getIterator()));
                rewriter.mergeBlocks(&lazy_block, target, llvm::None);

                rewriter.eraseOp(lazy_op);

                return res;
            }

            logical_result matchAndRewrite(
                LOp op, adaptor_t ops,
                conversion_rewriter &rewriter) const override
            {
                auto curr_block = rewriter.getBlock();
                auto rhs_block = curr_block->splitBlock(op);
                auto end_block = rhs_block->splitBlock(op);

                auto lhs_res = insert_lazy_to_block(ops.getLhs().getDefiningOp(), curr_block, rewriter);
                auto rhs_res = insert_lazy_to_block(ops.getRhs().getDefiningOp(), rhs_block, rewriter);

                auto lhs_res_type = lhs_res.getType();

                rewriter.setInsertionPointToEnd(curr_block);
                auto zero = rewriter.create< LLVM::ConstantOp >(
                    op.getLoc(), lhs_res_type, rewriter.getIntegerAttr(lhs_res_type, 0)
                );

                auto cmp_lhs = rewriter.create< LLVM::ICmpOp >(
                    op.getLoc(), LLVM::ICmpPredicate::eq, lhs_res, zero
                );

                auto end_arg = end_block->addArgument(cmp_lhs.getType(), op.getLoc());

                if constexpr (short_on_true)
                {
                    rewriter.create< LLVM::CondBrOp >(
                        op.getLoc(), cmp_lhs, end_block, cmp_lhs.getResult(), rhs_block, llvm::None
                    );
                } else {
                    rewriter.create< LLVM::CondBrOp >(
                        op.getLoc(), cmp_lhs, rhs_block, llvm::None, end_block, cmp_lhs.getResult()
                    );
                }

                rewriter.setInsertionPointToEnd(rhs_block);
                auto cmp_rhs = rewriter.create< LLVM::ICmpOp >(
                    op.getLoc(), LLVM::ICmpPredicate::eq, rhs_res, zero);
                rewriter.create< LLVM::BrOp >(op.getLoc(), cmp_rhs.getResult(), end_block);

                rewriter.setInsertionPointToStart(end_block);
                auto zext = rewriter.create< LLVM::ZExtOp >(op.getLoc(),
                    op.getResult().getType(),
                    end_arg);
                rewriter.replaceOp(op, { zext });

                return logical_result::success();
            }
        };

        using bin_lop_conversions = util::type_list<
            lazy_bin_logical< core::BinLAndOp, false >,
            lazy_bin_logical< core::BinLOrOp, true >
        >;

    } //namespace pattern

    struct CoreToLLVMPass : ModuleConversionPassMixin< CoreToLLVMPass, CoreToLLVMBase > {
        using base = ModuleConversionPassMixin< CoreToLLVMPass, CoreToLLVMBase >;

        static conversion_target create_conversion_target(MContext &context) {
            conversion_target target(context);
            target.addIllegalDialect< vast::core::CoreDialect >();
            target.addLegalOp< core::LazyOp >();
            target.addLegalDialect< mlir::LLVM::LLVMDialect>();
            return target;
        }

        static void populate_conversions(rewrite_pattern_set &patterns) {
            populate_conversions_base<
                pattern::bin_lop_conversions
            >(patterns);
        }
    };

    std::unique_ptr< mlir::Pass > createCoreToLLVMPass() {
        return std::make_unique< CoreToLLVMPass >();
    }
} //namespace vast
