// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"

#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Util/TypeUtils.hpp"

#include "vast/Conversion/TypeConverters/TypeConverter.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"
#include "vast/Conversion/TypeConverters/HLToStd.hpp"

#include <unordered_map>
#include <ranges>

namespace vast
{
    namespace pattern
    {
        struct structs_to_llvm : conv::tc::mixins< structs_to_llvm >,
                                 conv::tc::identity_type_converter,
                                 conv::tc::HLAggregates< structs_to_llvm >
        {
            using base = conv::tc::identity_type_converter;

            mcontext_t &mctx;
            vast_module mod;


            structs_to_llvm(mcontext_t &mctx, vast_module mod)
                : mctx(mctx), mod(mod)
            {

                init();
                conv::tc::HLAggregates< structs_to_llvm >::init();
            }

            mlir_type int_type(unsigned, bool)
            {
                VAST_UNREACHABLE("Unexpected path hit when lowering structures to LLVM");
            }

            maybe_types_t convert_field_types(mlir_type t)
            {
                auto def = hl::definition_of(t, mod);

                // Nothing found, leave the structure opaque.
                if (!def)
                    return {};

                mlir::SmallVector< mlir_type, 4 > out;
                for (auto field_type : hl::field_types(*def))
                {
                    auto c = convert_type_to_type(field_type);
                    VAST_ASSERT(c);
                    out.push_back(*c);
                }
                return { std::move(out) };
            }

            auto convert_record()
            {
                // We need this prototype to handle recursive types.
                return [&](hl::RecordType t,
                           mlir::SmallVectorImpl< mlir_type > &out,
                           mlir::ArrayRef< mlir_type > stack) -> logical_result
                {
                    auto core = mlir::LLVM::LLVMStructType::getIdentified(t.getContext(),
                                                                          t.getName());
                    // Last element is `t`.
                    auto bt = stack.drop_back();
                    if (core.isOpaque() && std::ranges::find(bt, t) == bt.end())
                    {
                        if (auto body = convert_field_types(t))
                        {
                            // Multithreading may cause some issues?
                            auto status = core.setBody(*body, false);
                            VAST_ASSERT(mlir::succeeded(status));
                        }

                    }
                    out.push_back(core);
                    return mlir::success();
                };
            }

            maybe_types_t do_conversion(mlir_type type)
            {
                types_t out;
                if (mlir::succeeded(this->convertType(type, out)))
                    return std::move(out);

                return {};
            }

            // Must be last so we can return `auto` in the helpers.
            void init()
            {
                addConversion(convert_record());
            }
        };

        struct struct_type_replacer : conv::tc::type_converting_pattern< structs_to_llvm >
        {
            using base = conv::tc::type_converting_pattern< structs_to_llvm >;

            struct_type_replacer(structs_to_llvm &tc, mcontext_t *mctx)
                : base(tc, *mctx)
            {}

            logical_result matchAndRewrite(
                mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
                conversion_rewriter &rewriter
            ) const override {
                if (auto func_op = mlir::dyn_cast< hl::FuncOp >(op))
                    return replace(func_op, ops, rewriter);
                return replace(op, ops, rewriter);
            }
        };

        template< typename op_t >
        struct erase : operation_conversion_pattern< op_t >
        {
            using base = operation_conversion_pattern< op_t >;
            using base::base;

            logical_result matchAndRewrite(op_t op,
                                           typename op_t::Adaptor ops,
                                           conversion_rewriter &rewriter
            ) const override {
                rewriter.eraseOp(op);
                return mlir::success();
            }
        };


    } // namespace pattern

    struct HLStructsToLLVMPass : HLStructsToLLVMBase< HLStructsToLLVMPass >
    {
        void runOnOperation() override
        {
            // Because `hl.struct` does not have a type, we first must convert
            // all used types - after we are done all definitions can be deleted
            // safely.
            replace_types();
            erase_defs();
        }

        void replace_types()
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            trg.markUnknownOpDynamicallyLegal([&](auto op)
            {
                return !has_type_somewhere< hl::RecordType >(op);
            });

            pattern::structs_to_llvm tc(mctx, op);

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< pattern::struct_type_replacer >(tc, patterns.getContext());

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
                return signalPassFailure();
        }

        void erase_defs()
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            trg.addIllegalOp< hl::StructDeclOp >();

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< pattern::erase< hl::StructDeclOp > >(patterns.getContext());

            // First we build all required types.
            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
                return signalPassFailure();
        }

    };

} // namespace vast

std::unique_ptr< mlir::Pass > vast::createHLStructsToLLVMPass()
{
    return std::make_unique< HLStructsToLLVMPass >();
}
