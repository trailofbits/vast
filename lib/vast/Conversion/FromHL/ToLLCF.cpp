// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Terminator.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Conversion/Common/Block.hpp"
#include "vast/Conversion/Common/Rewriter.hpp"


#include "vast/Dialect/Core/CoreTraits.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "../PassesDetails.hpp"

namespace vast::conv
{
    namespace
    {
        static inline const char *tie_fail = "base_pattern::tie failed.";

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

        template< typename Bld >
        mlir::Block *inline_region_before( Bld &bld, mlir::Region &region, mlir::Block *before )
        {
            auto begin = &region.front();
            auto end   = &region.back();
            VAST_CHECK( begin == end, "Region has more than one block" );

            bld.inlineRegionBefore( region, before );
            return begin;
        }

        auto cond_yield( mlir::Block *block )
        {
            auto cond_yield = hard_terminator_t::get( *block ).cast< hl::CondYieldOp >();
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

        template< typename Fn, typename H, typename ... Args >
        auto apply( Fn &&fn, mlir::Operation *op )
        {
            if ( auto casted = llvm::dyn_cast< H >( op ) )
                return fn( casted );
            if constexpr ( sizeof ... ( Args ) == 0 )
                return fn( op );
            else
                return apply< Fn, Args ... >( std::forward< Fn >( fn ), op );
        }

        template< typename Fn, typename ... Args >
        auto apply( Fn &&fn, Operation *op, util::type_list< Args ... > )
        {
            return apply< Fn, Args ... >( std::forward< Fn >( fn ), op );
        }

    } // namespace

    namespace pattern
    {
        auto get_cond_yield( mlir::Block &block )
        {
            return terminator_t< hl::CondYieldOp >::get( block ).op();
        }

        // We do not use patterns, because for example `hl.continue` in for loop is kinda tricky
        // as it does not use the same "entrypoint" as the scope and uses increment region
        // instead.
        template< typename bld_t >
        struct handle_terminators
        {
            using result_t = mlir::LogicalResult;
            using maybe_result_t = std::optional< mlir::LogicalResult >;

            using maybe_op_t = std::optional< mlir::Operation * >;

            bld_t &bld;
            // `entry` - where to jump in case of `continue`.
            //         - `nullptr` means the entry block of the scope is used.
            // `exit` - where to jump in case of `break`
            //        - `nullptr` means the next block after scope is used.
            mlir::Block *entry;
            mlir::Block *exit;

            handle_terminators( bld_t &bld, mlir::Block *entry, mlir::Block *exit )
                : bld( bld ), entry( entry ), exit( exit )
            {}

            result_t run( mlir::Region &region )
            {
                // Go instructions by instruction.
                for ( auto &block : region )
                    if ( mlir::failed( run( block ) ) )
                        return mlir::failure();
                return mlir::success();
            }

            result_t run( mlir::Block &block )
            {
                for ( auto &op : block )
                    if ( mlir::failed( run( &op ) ) )
                        return mlir::failure();
                return mlir::success();
            }

            result_t run( mlir::Operation *op )
            {
                if ( mlir::failed( replace( op ) ) )
                    return mlir::failure();

                if ( starts_cf_scope( op ) )
                    return mlir::success();

                for ( auto &region : op->getRegions() )
                    if ( mlir::failed( run( region ) ) )
                        return mlir::failure();
                return mlir::success();
            }

            bool starts_cf_scope( mlir::Operation *op )
            {
                // TODO( conv:hltollcf ): Define & use some trait instead.
                // TODO( conv:hltollcf ): Missing ops.
                return mlir::isa< hl::ForOp >( op ) || mlir::isa< hl::WhileOp >( op );
            }

            // TODO( conv:hltollcf ): Refactor using wrapper once we have it finalized.
            maybe_op_t do_replace( hl::ContinueOp op )
            {
                auto g = mlir::OpBuilder::InsertionGuard( bld );
                bld.setInsertionPointAfter( op );
                if ( entry )
                    return bld.template create< ll::Br >( op.getLoc(), entry );
                return bld.template create< ll::ScopeRecurse >( op.getLoc() );
            }

            maybe_op_t do_replace( hl::ReturnOp op )
            {
                auto g = mlir::OpBuilder::InsertionGuard( bld );
                bld.setInsertionPointAfter( op );
                return bld.template create< ll::ReturnOp >( op.getLoc(), op.getResult() );
            }

            maybe_op_t do_replace( hl::BreakOp op )
            {
                auto g = mlir::OpBuilder::InsertionGuard( bld );
                bld.setInsertionPointAfter( op );
                if ( exit )
                    return bld.template create< ll::Br >( op.getLoc(), exit );
                return bld.template create< ll::ScopeRet >( op.getLoc() );
            }

            // We did not match, do nothing.
            maybe_op_t do_replace( mlir::Operation *op )
            {
                return {};
            }

            result_t replace( mlir::Operation *op )
            {
                auto dispatch = [this]( auto op )
                {
                    if ( auto replaced = do_replace( op ) )
                        bld.eraseOp( op );
                    return mlir::success();
                };

                using ops = util::make_list< hl::BreakOp, hl::ContinueOp, hl::ReturnOp >;
                return apply( dispatch, op, ops{} );
            }

        };

        template< typename op_t >
        struct base_pattern : OpConversionPattern< op_t >
        {
            using parent_t = OpConversionPattern< op_t >;
            using parent_t::parent_t;

            static logical_result tie( auto &&bld, auto loc,
                                       mlir::Block &from, mlir::Block &to )
            {
                if ( !empty( from ) && any_terminator_t::get( from ) )
                    return mlir::success();

                VAST_CHECK( &from != &to, "Cannot create self-loop." );
                bld.template make_at_end< ll::Br >( &from, loc, &to );
                return mlir::success();
            }

            // Returns `[ cond_yield, coerced operand of cond_yield ]`
            static auto fetch_cond_yield( auto &&bld, mlir::Block &cond_block )
            {
                auto cond_yield = get_cond_yield( cond_block );
                auto g = bld.guard();
                bld->setInsertionPointAfter( cond_yield );
                auto value = coerce_condition( cond_yield.getResult(), *bld );

                return std::make_tuple( cond_yield, value );
            }
        };

        struct if_op : base_pattern< hl::IfOp >
        {
            using parent_t = base_pattern< hl::IfOp >;
            using parent_t::parent_t;

            mlir::LogicalResult matchAndRewrite(
                hl::IfOp op,
                hl::IfOp::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto bld = rewriter_wrapper_t( rewriter );

                auto [ original_block, tail_block ] = split_at_op( op, rewriter );
                VAST_CHECK( original_block && tail_block,
                            "Failed extraction of ifop into block." );

                auto cond_block = inline_region_before( rewriter,
                                                        op.getCondRegion(), tail_block );
                auto cond_yield_op = cond_yield( cond_block );
                auto cond_value    = coerce_yield( cond_yield_op, rewriter );

                auto false_block = [ &, tail_block = tail_block ] {
                    if ( op.hasElse() )
                        return inline_region_before( rewriter, op.getElseRegion(), tail_block );
                    return tail_block;
                }();

                auto true_block = inline_region_before( rewriter,
                                                        op.getThenRegion(), tail_block );

                bld.make_at_end< ll::CondBr >( cond_block,
                                               op.getLoc(), cond_value,
                                               true_block, false_block );
                rewriter.eraseOp( cond_yield_op );


                VAST_PATTERN_CHECK(parent_t::tie( bld, op.getLoc(), *true_block, *tail_block ),
                                   tie_fail);
                if ( false_block != tail_block )
                {
                    VAST_PATTERN_CHECK(parent_t::tie( bld, op.getLoc(),
                                                      *false_block, *tail_block ),
                                       tie_fail);
                }

                rewriter.mergeBlocks( cond_block, original_block, std::nullopt );

                // If only operation in a block is `hl.if` the way we split it
                // it can happen we end up with an empty block. For example
                // `hl.scope { hl.if { ... } }.
                // We therefore need to emit a dummy terminator to satisfy `mlir::Block`
                // verification.
                // We are using any_terminator as it can have for example `hl.return`
                // or other soft terminator that will get eliminated in this pass.
                if ( !any_terminator_t::has( *tail_block ) )
                {
                    bld.guarded_at_end( tail_block, [&](){
                        bld->template create< ll::ScopeRet >( op.getLoc() );
                    });
                }
                rewriter.eraseOp( op );

                return mlir::success();
            }

            static void legalize( conversion_target &trg )
            {
                trg.addIllegalOp< hl::IfOp >();
            }

        };

        struct while_op : base_pattern< hl::WhileOp >
        {
            using parent_t = base_pattern< hl::WhileOp >;
            using parent_t::parent_t;

            mlir::LogicalResult matchAndRewrite(
                hl::WhileOp op,
                hl::WhileOp::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto bld = rewriter_wrapper_t( rewriter );

                auto scope = rewriter.create< ll::Scope >( op.getLoc() );
                auto scope_entry = rewriter.createBlock( &scope.getBody() );

                auto &cond_region = op.getCondRegion();
                auto &body_region = op.getBodyRegion();

                if ( mlir::failed( handle_terminators( rewriter,
                                                       nullptr,
                                                       nullptr ).run( op.getBodyRegion() ) ) )
                {
                    return mlir::failure();
                }

                auto body_block = inline_region( rewriter,
                                                 body_region, scope.getBody() );

                // Condition block cannot be entry because entry block cannot have
                // predecessors and body block will jump to it.
                auto cond_block = inline_region( rewriter, cond_region, scope.getBody() );

                auto [ cond_yield, value ] = fetch_cond_yield( bld, *cond_block );
                VAST_CHECK( value, "Condition region yield unexpected type" );

                bld.make_at_end< ll::CondScopeRet >( cond_block,
                                                     op.getLoc(), *value, body_block );
                rewriter.eraseOp( cond_yield );

                VAST_PATTERN_CHECK(parent_t::tie( bld, op.getLoc(),
                                                   *scope_entry, *cond_block ),
                                   tie_fail);

                rewriter.eraseOp( op );
                return mlir::success();
            }

            static void legalize( conversion_target &trg )
            {
                trg.addIllegalOp< hl::WhileOp >();
            }
        };

        struct for_op : base_pattern< hl::ForOp >
        {
            using op_t = hl::ForOp;
            using parent_t = base_pattern< op_t >;
            using parent_t::parent_t;

            mlir::LogicalResult matchAndRewrite(
                op_t op,
                typename op_t::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto bld = rewriter_wrapper_t( rewriter );
                auto scope = rewriter.create< ll::Scope >( op.getLoc() );
                auto scope_entry = rewriter.createBlock( &scope.getBody() );

                auto make_inline_region = [&] (auto &&reg)
                {
                    return inline_region( rewriter, std::forward< decltype(reg) >(reg),
                                          scope.getBody() );
                };

                auto inc_block = make_inline_region( op.getIncrRegion() );

                if ( mlir::failed( handle_terminators( rewriter,
                                                       inc_block,
                                                       nullptr ).run( op.getBodyRegion() ) ) )
                {
                        return mlir::failure();
                }

                auto cond_block = make_inline_region( op.getCondRegion() );
                auto body_block = make_inline_region( op.getBodyRegion() );

                auto [ cond_yield, value ] = fetch_cond_yield( bld, *cond_block );
                VAST_PATTERN_CHECK( value, "Condition region yield unexpected type" );

                bld.make_at_end< ll::CondScopeRet >( cond_block,
                                                     op.getLoc(), *value, body_block );
                rewriter.eraseOp( cond_yield );

                auto mk_tie = [ & ]( auto &from, auto &to )
                {
                    return parent_t::tie( bld, op.getLoc(), from, to );
                };

                VAST_PATTERN_CHECK( mk_tie( *scope_entry, *cond_block ), tie_fail );
                VAST_PATTERN_CHECK( mk_tie( *body_block, *inc_block ), tie_fail );
                VAST_PATTERN_CHECK( mk_tie( *inc_block, *cond_block ), tie_fail );

                rewriter.eraseOp( op );

                return mlir::success();
            }

            static void legalize( conversion_target &trg )
            {
                trg.addIllegalOp< hl::ForOp >();
            }
        };

        template< typename op_t, typename trg_t >
        struct replace : base_pattern< op_t >
        {
            using parent_t = base_pattern< op_t >;
            using parent_t::parent_t;


            mlir::LogicalResult matchAndRewrite(
                op_t op,
                typename op_t::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
            {
                rewriter.create< trg_t >( op.getLoc(), ops.getOperands() );
                rewriter.eraseOp( op );
                return mlir::success();
            }

            static void legalize( conversion_target &trg )
            {
                trg.addIllegalOp< op_t >();
                trg.addLegalOp< trg_t >();
            }
        };

        using cf_patterns = util::make_list<
              if_op
            , while_op
            , for_op
            , replace< hl::ReturnOp, ll::ReturnOp >
        >;

    } // namespace pattern

    struct HLToLLCF : ModuleConversionPassMixin< HLToLLCF, HLToLLCFBase >
    {
        using base = ModuleConversionPassMixin< HLToLLCF, HLToLLCFBase >;
        using config_t = typename base::config_t;

        static auto create_conversion_target( mcontext_t &mctx )
        {
            mlir::ConversionTarget trg(mctx);
            trg.addLegalDialect< ll::LowLevelDialect >();
            trg.addLegalDialect< hl::HighLevelDialect >();

            trg.addIllegalOp< hl::ContinueOp >();
            trg.addIllegalOp< hl::BreakOp >();
            trg.addIllegalOp< hl::ReturnOp >();

            trg.addLegalOp< mlir::cf::BranchOp >();
            trg.markUnknownOpDynamicallyLegal([](auto){ return true; });
            return trg;
        }

        static void populate_conversions(config_t &config)
        {
            base::populate_conversions_base< pattern::cf_patterns >(config);
        }

        void after_operation() override
        {
            auto clean_scopes = [&](ll::Scope scope)
            {
                mlir::IRRewriter rewriter{ &this->getContext() };
                // We really don't care if anything ws remove or not.
                std::ignore = mlir::eraseUnreachableBlocks(rewriter, scope.getBody());
            };
            this->getOperation().walk(clean_scopes);

            auto clean_functions = [&](hl::FuncOp fn)
            {
                mlir::IRRewriter rewriter{ &this->getContext() };
                // We really don't care if anything ws remove or not.
                std::ignore = mlir::eraseUnreachableBlocks(rewriter, fn.getBody());
            };
            this->getOperation().walk(clean_functions);
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createHLToLLCFPass()
{
    return std::make_unique< vast::conv::HLToLLCF >();
}
