// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Conversion/Common/Types.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeUtils.hpp"

#include "vast/Conversion/TypeConverters/DataLayout.hpp"
#include "vast/Conversion/TypeConverters/HLToStd.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include <algorithm>
#include <iostream>

namespace vast::hl
{
    using type_converter = conv::tc::high_level_to_std_type_converter;

    namespace pattern {
        using lower_type = conv::tc::type_converting_pattern< type_converter >;
    } // namespace pattern

    struct HLLowerTypesPass : HLLowerTypesBase< HLLowerTypesPass >
    {
        void runOnOperation() override {
            auto op    = this->getOperation();
            auto &mctx = this->getContext();

            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
            type_converter tc(dl_analysis.getAtOrAbove(op), mctx);

            mlir::ConversionTarget trg(mctx);
            auto is_legal = tc.get_is_type_conversion_legal();
            trg.markUnknownOpDynamicallyLegal(is_legal);

            mlir::RewritePatternSet patterns(&mctx);

            patterns.add< pattern::lower_type >(tc, mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };
} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createHLLowerTypesPass() {
    return std::make_unique< HLLowerTypesPass >();
}
