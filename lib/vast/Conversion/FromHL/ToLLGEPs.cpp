// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Symbols.hpp"

namespace vast {
    namespace {
        struct record_member_op : mlir::OpConversionPattern< hl::RecordMemberOp >
        {
            using op_t = hl::RecordMemberOp;
            using base = mlir::OpConversionPattern< op_t >;
            using base::base;

            logical_result matchAndRewrite(
                op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter
            ) const override {
                auto parent_type = ops.getRecord().getType();

                auto mod = op->getParentOfType< vast_module >();
                if (!mod) {
                    return mlir::failure();
                }

                auto def = hl::definition_of(parent_type, mod);
                if (auto struct_decl = mlir::dyn_cast_or_null< hl::StructDeclOp >(*def)) {
                    return lower(op, ops, rewriter, struct_decl);
                }
                if (auto union_decl = mlir::dyn_cast_or_null< hl::UnionDeclOp >(*def)) {
                    return lower(op, ops, rewriter, union_decl);
                }

                return mlir::failure();
            }

            logical_result lower(
                op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter,
                hl::StructDeclOp struct_decl
            ) const {
                auto idx = hl::field_idx(op.getName(), struct_decl);
                if (!idx) {
                    return mlir::failure();
                }

                return replace(op, ops, rewriter, *idx);
            }

            logical_result lower(
                op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter,
                hl::UnionDeclOp union_decl
            ) const {
                // After lowered, union will only have one member.
                return replace(op, ops, rewriter, 0);
            }

            logical_result replace(
                op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter, auto idx
            ) const {
                auto gep = rewriter.create< ll::StructGEPOp >(
                    op.getLoc(), op.getType(), ops.getRecord(), rewriter.getI32IntegerAttr(idx),
                    op.getNameAttr()
                );
                rewriter.replaceOp(op, gep);
                return mlir::success();
            }
        };

    } // namespace

    struct HLToLLGEPsPass : HLToLLGEPsBase< HLToLLGEPsPass >
    {
        void runOnOperation() override {
            auto op    = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            trg.markUnknownOpDynamicallyLegal([](auto) { return true; });
            trg.addIllegalOp< hl::RecordMemberOp >();

            mlir::RewritePatternSet patterns(&mctx);

            patterns.add< record_member_op >(&mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };
} // namespace vast

std::unique_ptr< mlir::Pass > vast::createHLToLLGEPsPass() {
    return std::make_unique< vast::HLToLLGEPsPass >();
}
