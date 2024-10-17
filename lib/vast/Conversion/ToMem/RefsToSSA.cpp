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
                auto var = core::symbol_table::lookup< core::var_symbol >(op, op.getName());

                VAST_CHECK(var, "Variable {0} not present in the symbol table.", op.getName());
                VAST_CHECK(mlir::isa< ll::Cell >(var), "Variable {0} is not a cell."
                    "Lower variable to cells before lowering of references.",
                    op.getName()
                );

                rewriter.replaceOp(op, var);
                return mlir::success();
            }

            static void legalize(conversion_target &trg) {
                trg.addDynamicallyLegalOp< hl::DeclRefOp >([] (hl::DeclRefOp op) {
                    auto var = core::symbol_table::lookup< core::var_symbol >(op, op.getName());
                    // Declarations with global storage are not cells to keep their init region
                    if (auto decl_storage = mlir::dyn_cast< core::DeclStorageInterface >(var)) {
                        return decl_storage.hasGlobalStorage();
                    }
                    return false;
                });
            }
        };

    } // namespace pattern

    struct RefsToSSAPass : ConversionPassMixin< RefsToSSAPass, RefsToSSABase >
    {
        using base = ConversionPassMixin< RefsToSSAPass, RefsToSSABase >;

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
