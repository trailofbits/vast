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

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"

#include "vast/Util/Symbols.hpp"
#include "vast/Util/DialectConversion.hpp"

#include <unordered_map>

namespace vast::hl
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
            using tc_t = util::tc::LLVMTypeConverter;

            tc_t &tc;

            template< typename ... Args >
            DoConversion( tc_t &tc, Args && ... args )
                : parent_t(std::forward< Args >(args) ...), tc(tc)
            {}

            DoConversion( const self_t & ) = default;
            DoConversion( self_t && ) = default;

            using types_t = std::vector< mlir::Type >;

            types_t collect_field_tys(hl::StructDeclOp op) const
            {
                std::vector< mlir::Type > out;
                if (op.fields().empty())
                    return out;

                for (auto &maybe_field : solo_block(op.fields()))
                {
                    auto field = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
                    VAST_ASSERT(field);
                    if (auto c = tc.convert_type_to_type(field.type()))
                        out.push_back(*c);
                    else
                        out.push_back(field.type());
                }
                return out;
            }

            // TODO(lukas): Re-implement using some more generic scoping mechanism?
            std::vector< hl::TypeDeclOp > fetch_decls(hl::StructDeclOp op) const
            {
                std::vector< hl::TypeDeclOp > out;
                auto module_op = op->getParentOfType< mlir::ModuleOp >();
                for (auto &x : solo_block(module_op.body()))
                {
                    if (auto type_decl = mlir::dyn_cast< hl::TypeDeclOp >(x);
                        type_decl && type_decl.name() == op.name())
                    {
                        out.push_back(type_decl);
                    }
                }
                return out;
            }

            mlir::Type make_struct_type(mlir::MLIRContext &mctx,
                                        const types_t field_types,
                                        llvm::StringRef name) const
            {
                VAST_ASSERT(!name.empty());
                auto core = mlir::LLVM::LLVMStructType::getIdentified(&mctx, name);
                auto res = core.setBody(field_types, false);
                VAST_ASSERT(mlir::succeeded(res));
                return core;
            }

            mlir::LogicalResult convert()
            {
                auto field_tys = collect_field_tys(op);
                auto name = op.name();
                auto trg_ty = make_struct_type(*rewriter.getContext(), field_tys, name);

                rewriter.create< hl::TypeDefOp >(
                        op.getLoc(), op.name(), trg_ty);

                auto type_decls = fetch_decls(op);
                for (auto x : type_decls)
                    rewriter.eraseOp(x);

                rewriter.eraseOp(op);
                return mlir::success();
            }

        };

        using struct_decl_op = util::TypeConvertingPattern<
            hl::StructDeclOp, util::tc::LLVMTypeConverter, DoConversion
        >;

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
            util::tc::FullLLVMTypeConverter type_converter(&mctx, llvm_options, &dl_analysis);

            patterns.add< pattern::struct_decl_op >(type_converter, patterns.getContext());

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
                return signalPassFailure();

        }
    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createHLStructsToLLVMPass()
{
    return std::make_unique< HLStructsToLLVMPass >();
}
