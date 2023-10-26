// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
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

#include "vast/Util/Maybe.hpp"

#include "vast/Util/Symbols.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/TypeUtils.hpp"

#include "vast/Conversion/TypeConverters/TypeConverter.hpp"
#include "vast/Conversion/TypeConverters/LLVMTypeConverter.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include <unordered_map>

namespace vast
{

    namespace pattern
    {
        template< typename T >
        struct DoConversion {};

        template<>
        struct DoConversion< hl::StructDeclOp > : util::State< hl::StructDeclOp >
        {
            using parent_t = util::State< hl::StructDeclOp >;
            using self_t = DoConversion< hl::StructDeclOp >;

            conv::tc::LLVMTypeConverter &tc;

            template< typename... Args >
            DoConversion(conv::tc::LLVMTypeConverter &tc, Args &&...args)
                : parent_t(std::forward< Args >(args)...), tc(tc)
            {}

            DoConversion(const self_t &) = default;
            DoConversion(self_t &&)      = default;

            std::vector< mlir::Type > convert_field_types()
            {
                std::vector< mlir::Type > out;
                for (auto field_type : field_types(op))
                {
                    auto c = tc.convert_type_to_type(field_type);
                    VAST_ASSERT(c);
                    out.push_back(*c);
                }
                return out;
            }


            mlir_type convert_struct_type()
            {
                auto core = mlir::LLVM::LLVMStructType::getIdentified(op.getContext(),
                                                                      op.getName());
                auto status = core.setBody(convert_field_types(), false);
                VAST_ASSERT(mlir::succeeded(status));
                return core;

            }

            mlir::LogicalResult convert()
            {
                auto llvm_struct = convert_struct_type();
                rewriter.create< hl::TypeDefOp >(op.getLoc(), op.getName(), llvm_struct);
                // TODO(conv:hl-structs-to-llvm): Investigate if this is still required.
                conv::rewriter_wrapper_t(rewriter).safe_erase(hl::type_decls(op));

                rewriter.eraseOp(op);
                return mlir::success();
            }

        };

        using struct_decl_op = util::TypeConvertingPattern<
            hl::StructDeclOp, conv::tc::LLVMTypeConverter, DoConversion
        >;

        struct structs_to_llvm : conv::tc::mixins< structs_to_llvm >,
                                 conv::tc::base_type_converter
        {
            using base = conv::tc::base_type_converter;

            mcontext_t &mctx;

            structs_to_llvm(mcontext_t &mctx) : mctx(mctx)
            {
                addConversion([&](hl::RecordType t) { return convert(t); });
            }

            maybe_type_t convert(hl::RecordType t)
            {
                return mlir::LLVM::LLVMStructType::getIdentified(t.getContext(),
                                                                 t.getName());
            }

            maybe_types_t do_conversion(mlir_type type)
            {
                types_t out;
                if (mlir::succeeded(this->convertType(type, out)))
                    return std::move(out);

                return {};
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

    } // namespace pattern

    struct HLStructsToLLVMPass : HLStructsToLLVMBase< HLStructsToLLVMPass >
    {
        void runOnOperation() override
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            trg.addIllegalOp< hl::StructDeclOp >();
            // TODO(lukas): Why is this needed?
            trg.addLegalOp< hl::TypeDefOp >();
            trg.markUnknownOpDynamicallyLegal([](auto) { return true; });


            mlir::RewritePatternSet patterns(&mctx);

            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();

            mlir::LowerToLLVMOptions llvm_options{ &mctx };
            conv::tc::FullLLVMTypeConverter type_converter(&mctx, llvm_options, &dl_analysis);

            patterns.add< pattern::struct_decl_op >(type_converter, patterns.getContext());

            // First we build all required types.
            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
                return signalPassFailure();

            // Now we have all types in the context so we can go replacing.
            return replace_types();
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

            pattern::structs_to_llvm tc(mctx);

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< pattern::struct_type_replacer >(tc, patterns.getContext());

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
