// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Symbols.hpp"

namespace vast::hl
{
    namespace
    {
        mlir::Block &solo_block(mlir::Region &region)
        {
            VAST_ASSERT(region.hasOneBlock());
            return *region.begin();
        }
    }

    namespace pattern
    {
        template< typename T >
        struct DoConversion {};

        template< typename Op >
        struct State
        {
            using rewriter_t = mlir::ConversionPatternRewriter;

            Op op;
            typename Op::Adaptor operands;
            rewriter_t &rewriter;

            State(Op op, typename Op::Adaptor operands, rewriter_t &rewriter)
                : op(op), operands(operands), rewriter(rewriter)
            {}
        };


        template< typename Op >
        struct BasePattern : mlir::OpConversionPattern< Op >
        {
            using parent_t = mlir::OpConversionPattern< Op >;
            using operation_t = Op;
            using parent_t::parent_t;

            mlir::LogicalResult matchAndRewrite(
                    operation_t op,
                    typename operation_t::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                return DoConversion< operation_t >(op, ops, rewriter).convert();
            }
        };


        template<>
        struct DoConversion< hl::RecordMemberOp > : State< hl::RecordMemberOp >
        {
            using State< hl::RecordMemberOp >::State;

            std::optional< StructDeclOp > get_def(auto op, hl::NamedType named_type) const
            {
                std::optional< StructDeclOp > out;
                auto yield = [&](auto candidate)
                {
                    auto op = candidate.getOperation();
                    if (auto struct_decl = mlir::dyn_cast< hl::StructDeclOp >(op))
                        if (struct_decl.name() == named_type.getName().getName())
                        {
                            VAST_ASSERT(!out);
                            out = struct_decl;
                        }
                };

                auto tu = op->template getParentOfType< mlir::ModuleOp >();
                if (!tu)
                    return {};

                util::symbols(tu, yield);
                return out;
            }

            std::optional< std::size_t > get_idx(auto name, hl::StructDeclOp decl) const
            {
                std::size_t idx = 0;
                for (auto &maybe_field : solo_block(decl.fields()))
                {
                    auto field = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
                    if (!field)
                        return {};
                    if (field.name() == name)
                        return { idx };
                    ++idx;
                }
                return {};
            }

            hl::NamedType fetch_record_type(mlir::Type type)
            {
                auto unwrap = [](auto t)
                {
                    return t.getElementType().template dyn_cast< hl::NamedType >();
                };

                // TODO(lukas): Rework if we need to handle more cases, probably use
                //              type switch.
                if (auto x = type.dyn_cast< hl::LValueType >())
                    return unwrap(x);
                if (auto x = type.dyn_cast< hl::PointerType >())
                    return unwrap(x);
                return {};

            }

            mlir::LogicalResult convert()
            {
                auto parent_type = this->operands.record().getType();

                auto as_named_type = fetch_record_type(parent_type);
                if (!as_named_type)
                    return mlir::failure();

                auto maybe_def = get_def(op, as_named_type);
                if (!maybe_def)
                    return mlir::failure();

                auto raw_idx = get_idx(op.name(), *maybe_def);
                if (!raw_idx)
                    return mlir::failure();

                auto loc = this->op.getLoc();
                auto gep = rewriter.create< ll::StructGEPOp >(
                        loc,
                        this->op.getType(),
                        this->operands.record(),
                        rewriter.getI32IntegerAttr(*raw_idx),
                        this->op.nameAttr());
                rewriter.replaceOp( op, { gep } );

                return mlir::success();
            }

        };

        using record_member_op = BasePattern< hl::RecordMemberOp >;

    } // namespace pattern

    struct HLToLLGEPsPass : HLToLLGEPsBase< HLToLLGEPsPass >
    {
        void runOnOperation() override
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            trg.markUnknownOpDynamicallyLegal( [](auto) { return true; } );
            trg.addIllegalOp< hl::RecordMemberOp >();

            mlir::RewritePatternSet patterns(&mctx);

            patterns.add< pattern::record_member_op >(&mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
                return signalPassFailure();
        }
    };
}


std::unique_ptr< mlir::Pass > vast::hl::createHLToLLGEPsPass()
{
    return std::make_unique< vast::hl::HLToLLGEPsPass >();
}
