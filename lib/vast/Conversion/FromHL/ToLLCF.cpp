// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Passes.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Terminator.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "../PassesDetails.hpp"

namespace vast
{
    namespace
    {
        auto coerce_condition(auto op, mlir::ConversionPatternRewriter &rewriter)
        -> std::optional< mlir::Value >
        {
            auto int_type = op.getType().template dyn_cast< mlir::IntegerType >();
            if (!int_type)
                return {};

            auto i1 =  mlir::IntegerType::get(op.getContext(), 1u, mlir::IntegerType::Signless);
            if (int_type == i1)
                return { op };

            auto coerced = rewriter.create< hl::ImplicitCastOp >(
                op.getLoc(), i1, op, hl::CastKind::IntegralCast);
            return { coerced };
        }

        template< typename Op, typename Builder >
        auto extract_as_block( Op op, Builder &bld )
            -> std::tuple< mlir::Block *, mlir::Block *, mlir::Block * >
        {
            auto it = mlir::Block::iterator( op );
            auto head = op->getBlock();

            auto body = bld.splitBlock( head, it );
            ++it;

            auto tail = bld.splitBlock( body, it );
            VAST_CHECK( head && body && tail, "Extract instruction as solo block failed." );
            return { head, body, tail };
        }

        template< typename Op, typename Builder >
        auto split_at_op( Op op, Builder &bld )
        {
            auto it = mlir::Block::iterator( op );
            auto head = op->getBlock();

            auto body = bld.splitBlock( head, it );

            return std::make_tuple( head, body );
        }

        auto guarded( auto &bld, auto &&fn )
        {
            auto g = mlir::OpBuilder::InsertionGuard( bld );
            return fn();
        }

        auto guarded_at_end( auto &bld, mlir::Block *block, auto &&fn )
        {
            auto g = mlir::OpBuilder::InsertionGuard( bld );
            bld.setInsertionPointToEnd( block );
            return fn();
        }

        template< typename Trg, typename Bld, typename ... Args >
        auto make_after_op( Bld &bld, Operation *op, Args && ... args )
        {
            mlir::OpBuilder::InsertionGuard guard( bld );
            bld.setInsertionPointAfter( op );
            return bld.template create< Trg >( std::forward< Args >( args ) ... );
        }

        template< typename Trg, typename Bld, typename ... Args >
        auto make_at_end( Bld &bld, mlir::Block *block, Args && ... args )
        {
            mlir::OpBuilder::InsertionGuard guard( bld );
            bld.setInsertionPointToEnd( block );
            return bld.template create< Trg >( std::forward< Args >( args ) ... );
        }

        template< typename Op, typename Bld >
        auto inline_cond_region( Op op, mlir::Region &region, Bld &bld, mlir::Block *before,
                                 mlir::Block *true_block, mlir::Block *false_block )
            -> mlir::Block *
        {
            auto begin = &region.front();
            auto end   = &region.back();

            VAST_CHECK( begin == end, "Condition region has more than one block" );

            // What if there is a hl.cond.yield sooner or multiple of those?
            auto cond_yield = get_terminator( *end ).cast< hl::CondYieldOp >();
            bld.inlineRegionBefore( region, before );

            VAST_CHECK( cond_yield, "Last block of condition region did not end with yield." );

            auto value = guarded( bld, [ & ] {
                bld.setInsertionPointAfter( cond_yield );
                return coerce_condition( cond_yield.getResult(), bld );
            });
            VAST_CHECK( value, "Condition region yield unexpected type" );

            guarded( bld, [ & ] {
                bld.setInsertionPointToEnd( end );
                bld.template create< ll::CondBr >( op.getLoc(), *value,
                                                   true_block, false_block );
                bld.eraseOp( cond_yield );
            });

            return begin;
        }

        template< typename Op, typename Bld >
        mlir::Block *inline_region( Op op, Bld &bld, mlir::Region &region, mlir::Block *before )
        {
            auto begin = &region.front();
            auto end   = &region.back();
            VAST_CHECK( begin == end, "Region has more than one block" );

            bld.inlineRegionBefore( region, before );
            return begin;
        }

        template< typename Op, typename Bld >
        mlir::Block *inline_region( Op op, Bld &bld, mlir::Region &region, mlir::Region &dest )
        {
            auto begin = &region.front();
            auto end   = &region.back();
            VAST_CHECK( begin == end, "Region has more than one block" );

            bld.inlineRegionBefore( region, dest, dest.end() );
            return begin;
        }

        auto terminate( auto &bld, mlir::Block *where, auto loc )
        {
            return guarded_at_end( bld, where, [ & ] {
                return bld.template create< ll::ScopeRet >( loc );
            });
        }

        auto cond_yield( mlir::Block *block )
        {
            auto cond_yield = get_terminator( *block ).cast< hl::CondYieldOp >();
            VAST_CHECK( cond_yield, "Block does not have a hl::CondYieldOp as terminator." );
            return cond_yield;
        }

        auto coerce_yield( hl::CondYieldOp op, mlir::ConversionPatternRewriter &bld )
        {
            return guarded( bld, [ & ] {
                bld.setInsertionPointAfter( op );
                auto maybe_val = coerce_condition( op.getResult(), bld );
                VAST_CHECK( maybe_val, "Coercion of yielded value failed" );
                return *maybe_val;
            });
        }

        bool is_hl_terminator( const auto &terminator )
        {
            return terminator.template is_one_of< hl::ReturnOp, hl::BreakOp, hl::ContinueOp >();
        }

    } // namespace

    namespace pattern
    {
        struct if_op : OpConversionPattern< hl::IfOp >
        {
            using parent_t = OpConversionPattern< hl::IfOp >;
            using parent_t::parent_t;

            mlir::LogicalResult matchAndRewrite(
                    hl::IfOp op,
                    hl::IfOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                //auto [ head, body, tail ] = extract_as_block( op, rewriter );
                auto [ original_block, tail_block ] = split_at_op( op, rewriter );
                VAST_CHECK( original_block && tail_block,
                            "Failed extraction of ifop into block." );

                auto cond_block = inline_region( op, rewriter, op.getCondRegion(), tail_block );
                auto cond_yield_op = cond_yield( cond_block );
                auto cond_value    = coerce_yield( cond_yield_op, rewriter );

                auto false_block = [ &, tail_block = tail_block ] {
                    if ( op.hasElse() )
                        return inline_region( op, rewriter, op.getElseRegion(), tail_block );
                    return tail_block;
                }();

                auto true_block = inline_region( op, rewriter, op.getThenRegion(), tail_block );

                make_at_end< ll::CondBr >( rewriter, cond_block, op.getLoc(),
                                           cond_value,
                                           true_block, false_block );
                rewriter.eraseOp( cond_yield_op );

                auto tie = [ & ]( auto from, auto to )
                {
                    VAST_CHECK( from != to, "Emitting branch would create self loop in if." );
                    guarded( rewriter, [ & ] {
                        rewriter.setInsertionPointToEnd( from );
                        rewriter.template create< ll::Br >( op.getLoc(), to );
                    } );
                };

                tie( true_block, tail_block );
                if ( false_block != tail_block )
                    tie( false_block, tail_block );

                rewriter.mergeBlocks( cond_block, original_block, llvm::None );

                rewriter.eraseOp( op );

                return mlir::success();
            }

            static void legalize( conversion_target &trg )
            {
                trg.addIllegalOp< hl::IfOp >();
            }

        };

        struct while_op : OpConversionPattern< hl::WhileOp >
        {
            using parent_t = OpConversionPattern< hl::WhileOp >;
            using parent_t::parent_t;

            mlir::LogicalResult matchAndRewrite(
                    hl::WhileOp op,
                    hl::WhileOp::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto scope = rewriter.create< ll::Scope >( op.getLoc() );
                auto scope_entry = rewriter.createBlock( &scope.body() );

                //auto [ head, body, tail ] = extract_as_block( op, rewriter );
                auto &cond_region = op.getCondRegion();
                auto &body_region = op.getBodyRegion();

                auto body_block = inline_region( op, rewriter,
                                                 body_region, scope.body() );

                // Condition block cannot be entry because entry block cannot have
                // predecessors and body block will jump to it.
                auto cond_block = inline_region( op, rewriter, cond_region, scope.body() );

                auto cond_yield = get_terminator( *cond_block ).cast< hl::CondYieldOp >();
                auto value = guarded( rewriter, [ & ] {
                    rewriter.setInsertionPointAfter( cond_yield );
                    return coerce_condition( cond_yield.getResult(), rewriter );
                });
                VAST_CHECK( value, "Condition region yield unexpected type" );

                guarded( rewriter, [ & ] {
                    rewriter.setInsertionPointToEnd( cond_block );
                    rewriter.template create< ll::CondScopeRet >( op.getLoc(), *value,
                                                                  body_block );
                    rewriter.eraseOp( cond_yield );
                });

                auto tie = [ & ]( auto from, auto to )
                {
                    guarded( rewriter, [ & ] {
                        rewriter.setInsertionPointToEnd( from );
                        rewriter.template create< ll::Br >( op.getLoc(), to );
                    } );
                };

                tie( scope_entry, cond_block );
                terminate( rewriter, body_block, op.getLoc() );

                rewriter.eraseOp( op );
                return mlir::success();
            }

            static void legalize( conversion_target &trg )
            {
                trg.addIllegalOp< hl::WhileOp >();
            }
        };

        using all = util::make_list< if_op, while_op >;

    } // namespace pattern

    struct HLToLLCF : ModuleConversionPassMixin< HLToLLCF, HLToLLCFBase >
    {
        using base = ModuleConversionPassMixin< HLToLLCF, HLToLLCFBase >;

        static auto create_conversion_target( MContext &mctx )
        {
            mlir::ConversionTarget trg(mctx);
            trg.addLegalDialect< ll::LowLevelDialect >();
            trg.addLegalDialect< hl::HighLevelDialect >();

            trg.addLegalOp< mlir::cf::BranchOp >();
            trg.addIllegalOp< hl::IfOp >();
            trg.addIllegalOp< hl::WhileOp >();
            trg.markUnknownOpDynamicallyLegal([](auto){ return true; });
            return trg;
        }

        static void populate_conversions( rewrite_pattern_set &patterns )
        {
            base::populate_conversions< pattern::all >( patterns );
        }
    };

    std::unique_ptr< mlir::Pass > createHLToLLCFPass()
    {
        return std::make_unique< HLToLLCF >();
    }
} // namespace vast
