// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

namespace vast::conv {

    struct VarsToCellsPass : ModuleConversionPassMixin< VarsToCellsPass, VarsToCellsBase >
    {
        using base = ModuleConversionPassMixin< VarsToCellsPass, VarsToCellsBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            conversion_target target(mctx);
            return target;
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createVarsToCellsPass() {
    return std::make_unique< vast::conv::VarsToCellsPass >();
}
