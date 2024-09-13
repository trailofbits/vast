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

namespace vast::conv {

    namespace pattern {

        struct ref_to_ssa : operation_conversion_pattern< hl::DeclRefOp >
        {
            using base = operation_conversion_pattern< hl::DeclRefOp >;
            using base::base;

            using adaptor_t = hl::DeclRefOp::Adaptor;

            logical_result matchAndRewrite(
                hl::DeclRefOp op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto st = core::get_effective_symbol_table_for< core::var_symbol >(op);
                VAST_CHECK(st, "No effective symbol table found for variable reference resolution.");

                auto var = st->lookup< core::var_symbol >(op.getName());
                VAST_CHECK(var, "Variable {} not present in the symbol table.", op.getName());
                VAST_CHECK(mlir::isa< ll::Cell >(var), "Variable {} is not a cell."
                    "Lower variable to cells before lowering of references.",
                    op.getName()
                );

                rewriter.replaceOp(op, var);
                return mlir::success();
            }
        };

    } // namespace pattern

    struct RefsToSSAPass : ModuleConversionPassMixin< RefsToSSAPass, RefsToSSABase >
    {
        using base = ModuleConversionPassMixin< RefsToSSAPass, RefsToSSABase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::ref_to_ssa >(cfg);
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createRefsToSSAPass() {
    return std::make_unique< vast::conv::RefsToSSAPass >();
}
