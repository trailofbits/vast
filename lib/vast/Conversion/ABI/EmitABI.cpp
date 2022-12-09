// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include <mlir/Rewrite/PatternApplicator.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/ABI/ABI.hpp"
#include "vast/ABI/Driver.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"

#include "vast/Dialect/ABI/ABIOps.hpp"

#include <iostream>

namespace vast
{
    std::string add_abi_prefix(std::string name)
    {
        return "vast.abi." + std::move(name);
    }

    bool has_abi_prefix(const std::string &name)
    {
        return llvm::StringRef(name).startswith("vast.abi");
    }

    // TODO(abi): Implement.
    bool is_abi_twin(mlir::func::FuncOp func) { return true; }

    struct TypeConverter : util::TCHelpers< TypeConverter >, mlir::TypeConverter
    {
        TypeConverter(const mlir::DataLayout &dl, MContext &mctx)
            : dl(dl), mctx(mctx)
        {}

        const mlir::DataLayout &dl;
        MContext &mctx;
    };

    namespace pattern
    {
        struct func_type : OpConversionPattern< mlir::func::FuncOp >
        {
            using Base = OpConversionPattern< mlir::func::FuncOp >;
            using Op = mlir::func::FuncOp;

            TypeConverter &tc;

            func_type(TypeConverter &tc, MContext &mctx)
                : Base(tc, &mctx), tc(tc)
            {}


            mlir::LogicalResult matchAndRewrite(
                    Op op, typename Op::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {
                auto converted_type = abi::make_x86_64< Op >(op, tc.dl);
                llvm::dbgs() << converted_type.to_string() << "\n"; llvm::dbgs().flush();
                return mlir::failure();

            }
        };

    } // namespace pattern


    struct ABIfy : ABIfyBase< ABIfy >
    {
        void runOnOperation() override
        {
            auto &mctx = this->getContext();
            mlir::ModuleOp op = this->getOperation();

            mlir::ConversionTarget target(mctx);
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });
            auto should_transform = [&](mlir::func::FuncOp op) { return !is_abi_twin(op); };
            target.addDynamicallyLegalOp< mlir::func::FuncOp >(should_transform);

            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();

            auto type_converter = TypeConverter(dl_analysis.getAtOrAbove(op), mctx);

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< pattern::func_type >(type_converter, mctx);
            if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns))))
                return signalPassFailure();

        }
    };

} // namespace vast

std::unique_ptr< mlir::Pass > vast::createABIfyPass()
{
    return std::make_unique< vast::ABIfy >();
}
