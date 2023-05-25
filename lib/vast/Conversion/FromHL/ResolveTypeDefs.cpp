// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"

#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

#include "../PassesDetails.hpp"

namespace vast::conv
{
    struct ResolveTypeDefs : ModuleConversionPassMixin< ResolveTypeDefs, ResolveTypeDefsBase >
    {
        using base = ModuleConversionPassMixin< ResolveTypeDefs, ResolveTypeDefsBase >;
        using config_t = typename base::config_t;

        static auto create_conversion_target( mcontext_t &mctx )
        {
            mlir::ConversionTarget trg(mctx);
            trg.markUnknownOpDynamicallyLegal([](auto){ return true; });
            return trg;
        }

        static void populate_conversions(config_t &config)
        {
            // TODO: fill;
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createResolveTypeDefsPass()
{
    return std::make_unique< vast::conv::ResolveTypeDefs >();
}
