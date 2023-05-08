// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include <mlir/Rewrite/PatternApplicator.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/ABI/ABI.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"

#include "vast/Dialect/ABI/ABIOps.hpp"

#include <iostream>
#include <unordered_map>

namespace vast
{
    namespace
    {

    } // namespace

    struct LowerABI : LowerABIBase< LowerABI >
    {
        void runOnOperation() override
        {
        }
    };

} // namespace vast

std::unique_ptr< mlir::Pass > vast::createLowerABIPass()
{
    return std::make_unique< vast::LowerABI >();
}
