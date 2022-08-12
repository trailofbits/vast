// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeConverter.hpp"

#include <iostream>

namespace vast::hl
{
    struct HLStructsToLLVMPass : HLStructsToLLVMBase< HLStructsToLLVMPass >
    {
        void runOnOperation() override
        {
            // TODO(lukas): Implement.
        }
    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createHLStructsToLLVMPass()
{
    return std::make_unique< HLStructsToLLVMPass >();
}
