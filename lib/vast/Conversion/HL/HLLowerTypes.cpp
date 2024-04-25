// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"

#include "vast/Conversion/Common/Types.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeUtils.hpp"

#include "vast/Conversion/TypeConverters/DataLayout.hpp"
#include "vast/Conversion/TypeConverters/HLToStd.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include <algorithm>
#include <iostream>

namespace vast::conv {
    using type_converter_t = conv::tc::HLToStd;

    namespace pattern {
        using lower_type = conv::tc::generic_type_converting_pattern< type_converter_t >;
    } // namespace pattern

    struct HLLowerTypesPass : HLLowerTypesBase< HLLowerTypesPass >
    {
        void runOnOperation() override {
            auto op    = this->getOperation();
            auto &mctx = this->getContext();

            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
            type_converter_t type_converter(dl_analysis.getAtOrAbove(op), mctx);

            mlir::ConversionTarget trg(mctx);
            auto is_legal = type_converter.get_is_type_conversion_legal();
            trg.markUnknownOpDynamicallyLegal(is_legal);

            mlir::RewritePatternSet patterns(&mctx);

            patterns.add< pattern::lower_type >(type_converter, mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };
} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createHLLowerTypesPass() {
    return std::make_unique< vast::conv::HLLowerTypesPass >();
}
