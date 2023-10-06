// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/TypeUtils.hpp"

#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "PassesDetails.hpp"

namespace vast::hl {
    namespace {
        using type_map = std::map< mlir_type, mlir_type >;

        namespace pattern {
            struct type_converter
                : tc::base_type_converter
                , tc::mixins< type_converter >
            {
                vast_module mod;

                type_converter(mcontext_t &mctx, vast_module mod)
                    : tc::base_type_converter(), mod(mod)
                {
                    addConversion([&](mlir_type t) { return this->convert(t); });
                }

                maybe_types_t do_conversion(mlir_type type) {
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

            struct resolve_typedef : generic_conversion_pattern
            {
                using base = generic_conversion_pattern;
                using base::base;

                type_converter &tc;

                resolve_typedef(type_converter &tc, mcontext_t &mctx)
                    : base(tc, mctx), tc(tc)
                {}

                logical_result rewrite(
                    hl::FuncOp fn, mlir::ArrayRef< mlir::Value > ops,
                    conversion_rewriter &rewriter
                ) const {
                    auto trg = tc.convert_type_to_type(fn.getFunctionType());
                    VAST_PATTERN_CHECK(trg, "Failed type conversion of, {0}", fn);

                    rewriter.updateRootInPlace(fn, [&]() {
                        fn.setType(*trg);
                        if (fn->getNumRegions() != 0) {
                            fixup_entry_block(fn.getBody());
                        }
                    });

                    return mlir::success();
                }

                logical_result matchAndRewrite(
                    mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
                    conversion_rewriter &rewriter
                ) const override {
                    // Special case for functions, it may be that we can unify it with
                    // the generic one.
                    if (auto fn = mlir::dyn_cast< hl::FuncOp >(op)) {
                        return rewrite(fn, ops, rewriter);
                    }

                    auto new_rtys = tc.convert_types_to_types(op->getResultTypes());
                    VAST_PATTERN_CHECK(new_rtys, "Type conversion failed in op {0}", *op);

                    auto do_change = [&]() {
                        for (std::size_t i = 0; i < new_rtys->size(); ++i) {
                            op->getResult(i).setType((*new_rtys)[i]);
                        }

                        if (op->getNumRegions() != 0) {
                            fixup_entry_block(op->getRegion(0));
                        }

                        // TODO unify with high level type conversion
                        mlir::AttrTypeReplacer replacer;
                        replacer.addReplacement(tc::convert_type_attr(tc));
                        replacer.recursivelyReplaceElementsIn(
                            op, true /* replace attrs */, false /* replace locs */, true /* replace types */
                        );
                    };

                    rewriter.updateRootInPlace(op, do_change);

                    return mlir::success();
                }

                void fixup_entry_block(mlir::Region &region) const {
                    if (region.empty()) {
                        return;
                    }

                    for (auto arg: region.front().getArguments()) {
                        auto trg = tc.convert_type_to_type(arg.getType());
                        VAST_PATTERN_CHECK(trg, "Type conversion failed: {0}", arg);
                        arg.setType(*trg);
                    }
                }
            };

            using all = util::make_list< resolve_typedef >;

        } // namespace pattern
    } // namespace

    struct ResolveTypeDefs : ModuleConversionPassMixin< ResolveTypeDefs, ResolveTypeDefsBase >
    {
        using base     = ModuleConversionPassMixin< ResolveTypeDefs, ResolveTypeDefsBase >;
        using config_t = typename base::config_t;

        static auto create_conversion_target(mcontext_t &mctx) {
            mlir::ConversionTarget trg(mctx);

            trg.markUnknownOpDynamicallyLegal([](operation op) {
                return !has_type_somewhere< hl::TypedefType >(op)
                    || mlir::isa< hl::TypeDefOp >(op);
            });

            return trg;
        }

        void runOnOperation() override {
            auto &mctx     = getContext();
            auto target    = create_conversion_target(mctx);
            vast_module op = getOperation();

            rewrite_pattern_set patterns(&mctx);

            auto tc = pattern::type_converter(mctx, op);
            patterns.template add< pattern::resolve_typedef >(tc, mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createResolveTypeDefsPass() {
    return std::make_unique< vast::hl::ResolveTypeDefs >();
}
