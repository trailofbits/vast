// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
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

namespace vast::hl
{
    namespace
    {
        using type_map = std::map< mlir_type, mlir_type >;

        namespace pattern
        {
            struct type_converter : mlir::TypeConverter,
                                    util::TCHelpers< type_converter >
            {
                using maybe_type = std::optional< mlir_type >;

                mcontext_t &mctx;
                vast_module mod;

                type_converter(mcontext_t &mctx, vast_module mod)
                    : mlir::TypeConverter(),
                      mctx(mctx),
                      mod(mod)
                {
                    addConversion([&](mlir_type t)
                    {
                        return this->convert(t);
                    });
                    addConversion([&](mlir::SubElementTypeInterface t)
                    {
                        return this->convert(t);
                    });
                }

                maybe_types_t do_conversion(mlir_type type)
                {
                    types_t out;
                    if (mlir::succeeded(this->convertTypes(type, out)))
                        return { std::move(out) };
                    return {};
                }

                // TODO(conv): This may need to be precomputed instead.
                maybe_type nested_type(mlir_type type)
                {
                    return hl::getBottomTypedefType(type, mod);
                }

                maybe_type convert(mlir_type type)
                {
                    return nested_type(type);
                }

                maybe_type convert(mlir::SubElementTypeInterface with_subelements)
                {
                    auto replacer = [&](hl::ElaboratedType elaborated)
                    {
                        return nested_type( elaborated );
                    };
                    return with_subelements.replaceSubElements(replacer);
                }
            };

            struct resolve_typedef : generic_conversion_pattern
            {
                using base = generic_conversion_pattern;
                using base::base;

                type_converter &tc;

                resolve_typedef(type_converter &tc, mcontext_t &mctx)
                    : base(tc, mctx),
                      tc(tc)
                {}


                logical_result rewrite(hl::FuncOp fn,
                                       mlir::ArrayRef< mlir::Value > ops,
                                       mlir::ConversionPatternRewriter &rewriter) const
                {
                    auto trg = tc.convert_type_to_type(fn.getFunctionType());
                    VAST_PATTERN_CHECK(trg, "Failed type conversion of, {0}", fn);

                    auto change = [&]()
                    {
                        fn.setType(*trg);
                        if (fn->getNumRegions() != 0)
                            fixup_entry_block(*fn->getRegions().begin());
                        // TODO(conv): Not yet sure how to ideally propagate this.
                        std::ignore = fix_attrs(fn.getOperation());
                    };

                    rewriter.updateRootInPlace(fn, change);
                    return mlir::success();
                }

                logical_result matchAndRewrite(
                        mlir::Operation *op,
                        mlir::ArrayRef< mlir::Value > ops,
                        mlir::ConversionPatternRewriter &rewriter) const override
                {
                    // Special case for functions, it may be that we can unify it with
                    // the generic one.
                    if (auto fn = mlir::dyn_cast< hl::FuncOp >(op))
                        return rewrite(fn, ops, rewriter);

                    auto new_rtys = tc.convert_types_to_types(op->getResultTypes());
                    VAST_PATTERN_CHECK(new_rtys, "Type conversion failed in op {0}", *op);

                    auto do_change = [&]()
                    {
                        for (std::size_t i = 0; i < new_rtys->size(); ++i)
                            op->getResult(i).setType((*new_rtys)[i]);

                        if (op->getNumRegions() != 0)
                            fixup_entry_block(*op->getRegions().begin());

                        // TODO(conv): Not yet sure how to ideally propagate this.
                        std::ignore = fix_attrs(op);
                    };

                    rewriter.updateRootInPlace(op, do_change);

                    return mlir::success();
                }

                void fixup_entry_block(mlir::Region &region) const
                {
                    if (region.empty())
                        return;
                    auto &block = *region.begin();
                    for (std::size_t i = 0; i < block.getNumArguments(); ++i)
                    {
                        auto arg = block.getArgument(i);
                        auto trg = tc.convert_type_to_type(arg.getType());
                        VAST_PATTERN_CHECK(trg, "Type conversion failed: {0}", arg);
                        arg.setType(*trg);
                    }
                }

                logical_result fix_attrs(mlir::Operation *op) const
                {
                    return util::AttributeConverter(*op->getContext(), tc).convert(op);
                }
            };

            using all = util::make_list< resolve_typedef >;

        } // namespace pattern
    } // namespace

    struct ResolveTypeDefs : ModuleConversionPassMixin< ResolveTypeDefs, ResolveTypeDefsBase >
    {
        using base = ModuleConversionPassMixin< ResolveTypeDefs, ResolveTypeDefsBase >;
        using config_t = typename base::config_t;

        static auto create_conversion_target(mcontext_t &mctx)
        {
            mlir::ConversionTarget trg(mctx);

            auto is_legal = [](operation op)
            {
                return !has_type_somewhere< hl::TypedefType >(op) ||
                       mlir::isa< hl::TypeDefOp >(op);
            };

            trg.markUnknownOpDynamicallyLegal(is_legal);
            return trg;
        }

        void runOnOperation() override
        {
            auto &mctx = getContext();
            auto target = create_conversion_target(mctx);
            vast_module op = getOperation();

            rewrite_pattern_set patterns(&mctx);

            auto tc = pattern::type_converter(mctx, op);
            patterns.template add< pattern::resolve_typedef >(tc, mctx);

            if (mlir::failed(mlir::applyPartialConversion(op,
                                                          target,
                                                          std::move(patterns))))
            {
                return signalPassFailure();
            }
        }

    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createResolveTypeDefsPass()
{
    return std::make_unique< vast::hl::ResolveTypeDefs >();
}
