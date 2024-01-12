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
#include "vast/Conversion/Common/Passes.hpp"

namespace vast::conv
{
    namespace
    {
        struct value_category_type_converter
            : tc::base_type_converter,
              tc::mixins< value_category_type_converter >
        {
            mlir::MLIRContext &mctx;

            // Do we need data layout? Probably not as everything should be explicit
            // type size wise.
            value_category_type_converter(mcontext_t &mctx)
                : mctx(mctx)
            {}

            using mixin_base = tc::mixins< value_category_type_converter >;
            using mixin_base::convert_type_to_type;
            using mixin_base::convert_type_to_types;

            template< typename T, typename... Args >
            auto make_aggregate_type(Args... args) {
                return [=](mlir_type elem) { return T::get(elem.getContext(), elem, args...); };
            }

            auto convert_lvalue_type() {
                return [&](hl::LValueType type) {
                    return Maybe(type.getElementType())
                        .and_then(convert_type_to_type())
                        .unwrap()
                        .and_then(make_aggregate_type< hl::PointerType >())
                        .template take_wrapped< maybe_type_t >();
                };
            }
        };

    } // namespace

    struct LowerValueCategoriesPass : LowerValueCategoriesBase< LowerValueCategoriesPass >
    {
        using base = LowerValueCategoriesBase< LowerValueCategoriesPass >;

        void runOnOperation() override
        {
            auto root = getOperation();
            auto &mctx = getContext();

            value_category_type_converter tc(mctx);

            mlir::ConversionTarget trg(mctx);

            mlir::RewritePatternSet patterns(&mctx);

            if (mlir::failed(mlir::applyPartialConversion(root, trg, std::move(patterns))))
                return signalPassFailure();
        }
    };
} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createLowerValueCategoriesPass()
{
    return std::make_unique< vast::conv::LowerValueCategoriesPass >();
}
