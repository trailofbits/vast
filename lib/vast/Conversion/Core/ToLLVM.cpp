// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/IRMapping.h>
VAST_UNRELAX_WARNINGS

#include <iterator>

#include "../PassesDetails.hpp"
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/TypeList.hpp"
#include "vast/Util/DialectConversion.hpp"

namespace vast
{
    namespace pattern
    {
        namespace LLVM = mlir::LLVM;


        template< typename Op >
        struct lazy_base : operation_conversion_pattern< Op >, llvm_pattern_utils
        {
            using base = operation_conversion_pattern< Op >;
            using base::base;

            auto lazy_into_block(
                Operation* lazy_op, block_t* target, conversion_rewriter &rewriter
            ) const -> std::tuple< mlir_value, block_t * > {
                auto &lazy_region = mlir::dyn_cast< core::LazyOp >(*lazy_op).getLazy();

                // In case `lazy_op` already has multiple blocks (nested lazy ops)
                // we need to return the block that does not yet have the control flow.
                // This will be the block with the yield, but if it is in the first block,
                // it will get merged.
                auto end_block = [&] {
                    if (&lazy_region.front() == &lazy_region.back())
                        return target;
                    return &lazy_region.back();
                }();

                // Last block should have hl.value.yield with the final value
                auto &yield = lazy_region.back().back();
                auto res = mlir::dyn_cast< hl::ValueYieldOp >(yield).getResult();
                rewriter.eraseOp(&yield);

                auto &first_block = lazy_region.front();
                // The rewriter API doesn't provide a call to insert into a selected block
                auto target_it = std::next(target->getIterator());
                rewriter.inlineRegionBefore(
                    lazy_region, *target->getParent(), target_it
                );
                rewriter.mergeBlocks(&first_block, target, std::nullopt);

                rewriter.eraseOp(lazy_op);

                return { res, end_block };
            }


            template< std::size_t count > requires (count > 1)
            auto split_into_blocks_at(operation op, mlir::Block *block) const {
                auto next = block->splitBlock(op);
                return std::tuple_cat(
                    std::make_tuple(next), split_into_blocks_at< count - 1 >(op, next));
            }

            template< std::size_t count > requires (count == 1)
            auto split_into_blocks_at(operation op, mlir::Block *block) const {
                return std::make_tuple(block->splitBlock(op));
            }

            auto mk_icmp_ne(auto &rewriter, auto loc, auto value) const {
                auto zero = iN(rewriter, loc, value.getType(), 0);
                return rewriter.template create< LLVM::ICmpOp >(
                    loc, LLVM::ICmpPredicate::ne, value, zero);
            }

            auto tie_block(
                auto &rewriter, auto loc, auto current_block, auto target_block, auto arg
            ) const {
                auto _ = insertion_guard(rewriter);
                auto current_it = current_block->getIterator();
                rewriter.setInsertionPointToEnd(&*current_it);
                if (arg)
                    rewriter.template create< LLVM::BrOp >(loc, arg, target_block);
                else
                    rewriter.template create< LLVM::BrOp >(loc, std::nullopt, target_block);
            }

            auto get_tie_block(auto &rewriter, auto loc, auto target_block) const {
                return [=, &rewriter](auto current_block, auto arg) {
                    return tie_block(rewriter, loc, current_block, target_block, arg);
                };
            }

            bool is_void(mlir_type t) const {
                return mlir::isa< mlir::LLVM::LLVMVoidType >(t);
            }

            auto add_argument(
                mlir::Block *block, auto type, auto loc
            ) const -> std::optional< mlir::BlockArgument > {
                if (is_void(type))
                    return std::nullopt;
                return std::make_optional(block->addArgument(type, loc));
            }
        };

        template< typename LOp, bool short_on_true >
        struct lazy_bin_logical : lazy_base< LOp >
        {
            using base = lazy_base< LOp >;
            using base::base;
            using adaptor_t = typename LOp::Adaptor;
            using base::lazy_into_block;
            using base::iN;

            void cond_br_lhs(
                conversion_rewriter &rewriter, auto loc, Value cond, block_t *rhs, block_t *end) const
            {
                if constexpr (short_on_true) {
                    rewriter.create< LLVM::CondBrOp >(loc, cond, end, cond, rhs, std::nullopt);
                } else {
                    rewriter.create< LLVM::CondBrOp >(loc, cond, rhs, std::nullopt, end, cond);
                }
            }

            logical_result matchAndRewrite(
                LOp op, adaptor_t ops, conversion_rewriter &rewriter) const override
            {
                /* Splitting the block at the place of the logical operation.
                 * It is divided into 3 parts:
                 *   1) the operations that happen before the logical operation, to this
                 *      part the evaluation of lhs is appended
                 *   2) empty block to which the evaluation of rhs is inserted
                 *   3) end block that recieves the evaluation of the logical operation
                 */
                auto curr_block = rewriter.getBlock();
                auto rhs_block = curr_block->splitBlock(op);
                auto end_block = rhs_block->splitBlock(op);

                auto [ lhs_res, _1 ] = lazy_into_block(ops.getLhs().getDefiningOp(), curr_block, rewriter);
                auto [ rhs_res, _2 ] = lazy_into_block(ops.getRhs().getDefiningOp(), rhs_block, rewriter);

                rewriter.setInsertionPointToEnd(curr_block);
                auto zero = iN(rewriter, op.getLoc(), lhs_res.getType(), 0);

                auto cmp_lhs = rewriter.create< LLVM::ICmpOp >(
                    op.getLoc(), LLVM::ICmpPredicate::ne, lhs_res, zero
                );

                // Block argument that recieves the result value
                auto end_arg = end_block->addArgument(cmp_lhs.getType(), op.getLoc());

                cond_br_lhs(rewriter, op.getLoc(), cmp_lhs, rhs_block, end_block);

                // In the case that the `rhs` region consists of multiple blocks (i.e.
                // it has already been lowered, since hl doesn't have blocks)
                // we need to insert the break to the last block of this region - which
                // is the region that is created by splitting the `end_block`.
                // e.g.: in a && (b || c) mlir first matches the `||` which creates
                // it's own cf blocks and then matches the `&&`
                auto rhs_end_it = std::prev(end_block->getIterator());
                rewriter.setInsertionPointToEnd(&*rhs_end_it);
                auto cmp_rhs = rewriter.create< LLVM::ICmpOp >(
                    op.getLoc(), LLVM::ICmpPredicate::ne, rhs_res, zero
                );
                rewriter.create< LLVM::BrOp >(op.getLoc(), cmp_rhs.getResult(), end_block);

                rewriter.setInsertionPointToStart(end_block);
                rewriter.replaceOpWithNewOp< LLVM::ZExtOp >(op, op.getResult().getType(), end_arg);

                return logical_result::success();
            }
        };

        struct select : lazy_base< core::SelectOp >
        {
            using op_t = core::SelectOp;
            using base = lazy_base<  op_t >;
            using base::base;

            using base::lazy_into_block;
            using base::iN;

            // This has 3 operands
            //  * condition - no work, we only emit a branch based on its value (icmp ne with `0`)
            //  * two lazy_ops - these will need to be inline as basic blocks (and keep in mind they
            //                   can already have some basic blocks inside due to nesting of selects
            //                   for example)
            logical_result matchAndRewrite(
                op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter) const override
            {
                auto curr_block = rewriter.getBlock();
                // New blocks in case we don't have blocks already inside (will get merged if not
                // needed).
                auto [true_block, false_block, end_block] = split_into_blocks_at< 3 >(op, curr_block);

                auto init_block = [&](auto operand, auto block) {
                    return lazy_into_block(operand.getDefiningOp(), block, rewriter);
                };

                // Value yielded by the block (can be empty if we are dealing with `void`) and
                // basic block to "continue" from (the one that had yield)
                auto [ true_res, true_end ] = init_block(ops.getThenRegion(), true_block);
                auto [ false_res, false_end ] = init_block(ops.getElseRegion(), false_block);

                rewriter.setInsertionPointToEnd(curr_block);
                auto cmp = mk_icmp_ne(rewriter, op.getLoc(), ops.getCond());

                // Block argument that receives the result value. Can be nothing in case of `void`.
                auto end_arg = add_argument(end_block, true_res.getType(), op.getLoc());

                rewriter.create< LLVM::CondBrOp >(
                    op.getLoc(), cmp, true_block, std::nullopt, false_block, std::nullopt);

                // Redirect control flow and if needed pass the argument along
                auto tie = get_tie_block(rewriter, op.getLoc(), end_block);
                tie(false_end, (end_arg) ? false_res : mlir_value{});
                tie(true_end, (end_arg) ? true_res : mlir_value{});

                rewriter.setInsertionPointToStart(end_block);

                // Not the cleanup of original op.
                auto trg_type = op.getResult(0).getType();
                // We may need to insert a cast.
                if (mlir::isa< mlir::IntegerType >(trg_type)) {
                    VAST_ASSERT(end_arg);
                    replace_with_trunc_or_ext(
                        op, *end_arg, end_arg->getType(), op.getResult(0).getType(), rewriter);
                // `void` value cannot be replaced, simply erase.
                } else if (is_void(trg_type)) {
                    rewriter.eraseOp(op);
                // Default - simply replace.
                } else {
                    VAST_ASSERT(end_arg);
                    rewriter.replaceOp(op, *end_arg);
                }
                return mlir::success();
            }
        };

        using lazy_op_conversions = util::type_list<
            lazy_bin_logical< core::BinLAndOp, false >,
            lazy_bin_logical< core::BinLOrOp, true >,
            select
        >;

    } //namespace pattern

    struct CoreToLLVMPass : ConversionPassMixin< CoreToLLVMPass, CoreToLLVMBase >
    {
        using base = ConversionPassMixin< CoreToLLVMPass, CoreToLLVMBase >;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);

            target.addIllegalDialect< vast::core::CoreDialect >();
            target.addLegalOp< core::LazyOp >();

            target.addLegalDialect< mlir::LLVM::LLVMDialect >();
            return target;
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::lazy_op_conversions >(cfg);
        }

        void run_after_conversion() {
            // Now we know that we need to get rid fo any remaining `llvm.mlir.zero` that
            // are of void type because they cannot be codegen'ed into LLVM IR.
            auto exec = [&](mlir::LLVM::ZeroOp op) {
                if (!llvm::isa< mlir::LLVM::LLVMVoidType >(op.getType()))
                    return;
                VAST_CHECK(op->getUsers().empty(), "{0} remains live after all conversions!", op);
                op->erase();
            };
            this->getOperation()->walk(exec);
        }
    };

    std::unique_ptr< mlir::Pass > createCoreToLLVMPass() {
        return std::make_unique< CoreToLLVMPass >();
    }
} //namespace vast
