// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"
#include "vast/Conversion/TypeConverters/HLToStd.hpp"

#include <ranges>

namespace rns = std::ranges;

namespace vast::conv {

    struct strip_param_lvalue_type_converter
        : tc::identity_type_converter
        , tc::mixins< strip_param_lvalue_type_converter >
        , tc::function_type_converter< strip_param_lvalue_type_converter >
    {
        mcontext_t &mctx;

        explicit strip_param_lvalue_type_converter(mcontext_t &mctx)
            : mctx(mctx)
        {
            tc::function_type_converter< strip_param_lvalue_type_converter >::init();
            addConversion([&](hl::LValueType type) {
                return Maybe(type.getElementType())
                    .and_then(convert_type_to_type())
                    .unwrap()
                    .template take_wrapped< maybe_type_t >();
            });
        }
    };

    namespace pattern {
        using strip_param_lvalue = tc::generic_type_converting_pattern<
            strip_param_lvalue_type_converter
        >;
    } // namespace pattern

    struct StripParamLValuesPass
        : TypeConvertingConversionPassMixin<
            StripParamLValuesPass,
            StripParamLValuesBase,
            strip_param_lvalue_type_converter
        >
    {
        using base = ConversionPassMixin< StripParamLValuesPass, StripParamLValuesBase >;

        static bool is_not_lvalue_type(mlir_type ty) {
            return !mlir::isa< hl::LValueType >(ty);
        }

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            conversion_target trg(mctx);

            trg.markUnknownOpDynamicallyLegal([] (operation op) {
                if (auto fn = mlir::dyn_cast< mlir::FunctionOpInterface >(op)) {
                    auto fty = mlir::cast< core::FunctionType >(fn.getFunctionType());
                    return rns::all_of(fty.getInputs(), is_not_lvalue_type);
                }
                return true;
            });

            return trg;
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::strip_param_lvalue >(cfg);
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createStripParamLValuesPass() {
    return std::make_unique< vast::conv::StripParamLValuesPass >();
}
