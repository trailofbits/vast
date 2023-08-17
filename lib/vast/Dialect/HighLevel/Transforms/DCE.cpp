// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

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
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{
    namespace
    {
        template< typename ... Args >
        bool is_one_of( mlir::Operation *op )
        {
            return ( mlir::isa< Args >( op ) || ... );
        }

    } // namespace

    struct DCE : DCEBase< DCE >
    {
        using base = DCEBase< DCE >;
        using operations_t = std::vector< mlir::Operation * >;

        operations_t to_erase;

        void simplify( mlir::Operation *root )
        {
            // If there is no reagion we have nothing to do.
            for ( auto &region : root->getRegions() )
                simplify( region );
        }

        void simplify( mlir::Region &region )
        {
            for ( auto &block : region )
                simplify( block );
        }

        bool is_terminator_like( mlir::Operation *op )
        {
            return is_one_of< hl::ReturnOp, hl::BreakOp, hl::ContinueOp >( op );
        }

        void simplify( mlir::Block &block )
        {
            auto breakpoint = simplify_until_terminator( block );
            while ( breakpoint != block.end() )
                to_erase.emplace_back( &*( breakpoint++ ) );
        }

        auto simplify_until_terminator( mlir::Block &block )
            -> mlir::Block::iterator
        {
            for ( auto it = block.begin(); it != block.end(); std::advance( it, 1 ) )
            {
                if ( is_terminator_like( &*it ) )
                    return std::next( it );
                simplify( &*it );
            }
            return block.end();
        }

        void runOnOperation() override
        {
            auto root = getOperation();

            simplify( root );
            // Due to dataflow dependencies last SSA values must be destroyed first.
            std::reverse( to_erase.begin(), to_erase.end() );
            for ( auto op : to_erase )
                op->erase();

            // This is what mlir codebase does.
            to_erase.clear();
        }
    };

    std::unique_ptr< mlir::Pass > createDCEPass()
    {
        return std::make_unique< DCE >();
    }
} // namespace vast::hl
