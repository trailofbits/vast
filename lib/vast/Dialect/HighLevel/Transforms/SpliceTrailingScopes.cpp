// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/IRMapping.h>

#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/Scopes.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{
    struct SpliceTrailingScopes : SpliceTrailingScopesBase< SpliceTrailingScopes >
    {
        using base = SpliceTrailingScopesBase< SpliceTrailingScopes >;

        using operations_t = std::vector< operation >;

        operations_t to_splice;

        void splice_trailing_scope(operation op)
        {
            auto scope = mlir::dyn_cast< hl::ScopeOp >(op);
            VAST_ASSERT(scope && "Op is not a scope!");

            auto parent = scope->getParentRegion();
            auto target = scope->getBlock();

            auto &body = scope.getBody();
            bool empty_body = body.empty();
            auto &start = body.front();

            scope->remove();

            parent->getBlocks().splice(target->getIterator(), body.getBlocks());

            auto &ops = target->getOperations();

            if (!empty_body)
            {
                ops.splice(ops.end(), start.getOperations());
                start.erase();
            }
            scope.erase();
        }

        void find(Block &block)
        {
            for (auto &op : block.getOperations())
                find(&op);
        }

        void find(Region &region)
        {
            for (auto &block : region.getBlocks())
                find(block);
        }

        void find(operation op)
        {
            if (is_trailing_scope(op))
                to_splice.emplace_back(op);
            for (auto &region : op->getRegions())
                find(region);
        }

        void runOnOperation() override
        {
            auto op = getOperation();
            find(op);
            std::reverse(to_splice.begin(), to_splice.end());
            for (auto op : to_splice)
                splice_trailing_scope(op);
        }
    };
} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createSpliceTrailingScopes()
{
    return std::make_unique< vast::hl::SpliceTrailingScopes >();
}
