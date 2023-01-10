// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/LowLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Symbols.hpp"

#include <iostream>

namespace vast::ll
{
    using mctx_t = mlir::MLIRContext;

    struct ToLLVMPass : ToLLVMBase< ToLLVMPass >
    {
        void runOnOperation() override;
    };

    void ToLLVMPass::runOnOperation()
    {
        auto &mctx = this->getContext();
        mlir::ModuleOp op = this->getOperation();


        mlir::ConversionTarget target(mctx);
        target.addIllegalDialect< ll::LowLevelDialect >();
        target.markUnknownOpDynamicallyLegal([](auto) { return true; });

        mlir::LowerToLLVMOptions llvm_options{ &mctx };
        llvm_options.useBarePtrCallConv = true;

        mlir::RewritePatternSet patterns(&mctx);

        if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns))))
            return signalPassFailure();
    }
} // namespace vast::ll


std::unique_ptr< mlir::Pass > vast::ll::createToLLVMPass()
{
    return std::make_unique< ToLLVMPass >();
}
