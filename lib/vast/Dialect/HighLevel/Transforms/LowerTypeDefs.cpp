// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Conversion/TypeConverters/DataLayout.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/TypeUtils.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "PassesDetails.hpp"

namespace vast::hl {
    namespace {
        using type_map = std::map< mlir_type, mlir_type >;

        namespace pattern {
            struct type_converter
                : conv::tc::base_type_converter
                , conv::tc::mixins< type_converter >
            {
                core::module mod;
                mcontext_t &mctx;

                type_converter(mcontext_t &mctx, core::module mod)
                    : conv::tc::base_type_converter(),
                      mod(mod), mctx(mctx)
                {
                    addConversion([&](mlir_type t) { return this->convert(t); });
                }

                maybe_types_t do_conversion(mlir_type type) const {
                    types_t out;
                    if (mlir::succeeded(this->convertTypes(type, out))) {
                        return { std::move(out) };
                    }
                    return {};
                }

                // TODO(conv): This may need to be precomputed instead.
                maybe_type_t nested_type(mlir_type type) {
                    return hl::getBottomTypedefType(type, mod);
                }

                maybe_type_t convert(mlir_type type) {
                    mlir::AttrTypeReplacer replacer;
                    replacer.addReplacement([this] (mlir_type t) {
                        return nested_type(t);
                    });
                    return replacer.replace(type);
                }
            };

            struct resolve_typedef
                : conv::tc::type_converting_pattern< type_converter >
            {
                using base = conv::tc::type_converting_pattern< type_converter >;
                using base::base;

                logical_result matchAndRewrite(
                    operation op, mlir::ArrayRef< mlir::Value > ops,
                    conversion_rewriter &rewriter
                ) const override {
                    auto status = base::matchAndRewrite(op, ops, rewriter);

                    if (mlir::isa< hl::TypeDefOp >(op))
                        rewriter.eraseOp(op);

                    return status;
                }
            };

        } // namespace pattern
    } // namespace

    struct LowerTypeDefs : ConversionPassMixin< LowerTypeDefs, LowerTypeDefsBase >
    {
        static auto create_conversion_target(mcontext_t &mctx) {
            mlir::ConversionTarget trg(mctx);

            trg.markUnknownOpDynamicallyLegal([](operation op) {
                return !has_type_somewhere< hl::TypedefType >(op);
            });

            trg.addIllegalOp< hl::TypeDefOp >();

            return trg;
        }

        void runOnOperation() override {
            auto &mctx  = getContext();
            auto target = create_conversion_target(mctx);
            auto op     = getOperation();

            rewrite_pattern_set patterns(&mctx);

            auto tc = pattern::type_converter(mctx, op);
            patterns.template add< pattern::resolve_typedef >(tc, mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createLowerTypeDefsPass() {
    return std::make_unique< vast::hl::LowerTypeDefs >();
}
