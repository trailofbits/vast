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
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

namespace vast::conv {

    namespace pattern {

        struct var_to_cell : operation_conversion_pattern< hl::VarDeclOp >
        {
            using base = operation_conversion_pattern< hl::VarDeclOp >;
            using base::base;

            using adaptor_t = hl::VarDeclOp::Adaptor;

            // Inline the region that is responsible for initialization
            //  * `rewriter` insert point is invalidated (although documentation of called
            //    methods does not state it, experimentally it is corrupted)
            //  * terminator is returned to be used & erased by caller.
            operation inline_init_region(auto src, auto &rewriter) const {
                auto &init_region = src.getInitializer();
                auto &init_block  = init_region.back();

                auto terminator = init_block.getTerminator();
                rewriter.inlineRegionBefore(init_region, src->getBlock());
                rewriter.inlineBlockBefore(&init_block, src.getOperation());
                return terminator;
            }

            logical_result matchAndRewrite(
                hl::VarDeclOp op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto cell = rewriter.create< ll::Cell >(op.getLoc(), op.getType(), op.getSymName(), op.getStorageClass(), op.getThreadStorageClass());

                if (auto &init = op.getInitializer(); !init.empty()) {
                    auto yield = inline_init_region(op, rewriter);
                    rewriter.setInsertionPointAfter(yield);
                    rewriter.create< ll::CellInit >(
                        yield->getLoc(), op.getType(), cell, yield->getOperand(0)
                    );
                    rewriter.eraseOp(yield);
                }

                rewriter.eraseOp(op);
                return mlir::success();
            }

            static void legalize(conversion_target &trg) {
                trg.addDynamicallyLegalOp< hl::VarDeclOp >([] (hl::VarDeclOp op) {
                    return !op.hasLocalStorage();
                });
                trg.addLegalOp< ll::Cell >();
                trg.addLegalOp< ll::CellInit >();
            }
        };

        struct param_to_cell : operation_conversion_pattern< hl::ParmVarDeclOp >
        {
            using base = operation_conversion_pattern< hl::ParmVarDeclOp >;
            using base::base;

            using adaptor_t = hl::ParmVarDeclOp::Adaptor;

            logical_result matchAndRewrite(
                hl::ParmVarDeclOp op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto param = op.getParam();
                auto type  = param.getType();
                auto loc   = op.getLoc();
                auto cell = rewriter.create< ll::Cell >(loc, type, op.getSymName(), core::StorageClass::sc_none, core::TSClass::tsc_none);
                rewriter.create< ll::CellInit >(loc, type, cell, param);
                rewriter.eraseOp(op);
                return mlir::success();
            }

            static void legalize(conversion_target &trg) {
                base::legalize(trg);
                trg.addLegalOp< ll::Cell >();
                trg.addLegalOp< ll::CellInit >();
            }
        };

    } // namespace pattern

    struct VarsToCellsPass : ConversionPassMixin< VarsToCellsPass, VarsToCellsBase >
    {
        using base = ConversionPassMixin< VarsToCellsPass, VarsToCellsBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            auto trg = conversion_target(mctx);
            // Block inlining might trigger legalization on some operations
            trg.addLegalDialect< ll::LowLevelDialect >();
            return trg;
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::var_to_cell >(cfg);
            base::populate_conversions< pattern::param_to_cell >(cfg);
        }
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createVarsToCellsPass() {
    return std::make_unique< vast::conv::VarsToCellsPass >();
}
