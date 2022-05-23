// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/PatternApplicator.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>

#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/raw_ostream.h>
VAST_UNRELAX_WARNINGS


#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

#include "PassesDetails.hpp"

namespace vast::hl
{
    class ToLLLVMIR : public mlir::LLVMTranslationDialectInterface
    {
      public:
        using Base = mlir::LLVMTranslationDialectInterface;
        using Base::Base;

        mlir::LogicalResult convertOperation(mlir::Operation *op, llvm::IRBuilderBase &irb,
                                             mlir::LLVM::ModuleTranslation &state) const final
        {
            return llvm::TypeSwitch< mlir::Operation *, mlir::LogicalResult >(op)
                .Case([&](hl::TypeDefOp) {
                    return mlir::success();
                })
                .Default([&](mlir::Operation *) {
                    return mlir::failure();
                });
        }
    };

    struct LLVMDump : LLVMDumpBase< LLVMDump >
    {
        void runOnOperation() override;
    };

    void LLVMDump::runOnOperation()
    {
        auto &mctx = this->getContext();
        mlir::ModuleOp op = this->getOperation();

        registerHLToLLVMIR(mctx);
        mlir::DialectRegistry registry;
        mlir::registerAllToLLVMIRTranslations(registry);
        mctx.appendDialectRegistry(registry);

        llvm::LLVMContext lctx;
        auto lmodule = mlir::translateModuleToLLVMIR(op, lctx);
        if (!lmodule)
            return signalPassFailure();


        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        mlir::ExecutionEngine::setupTargetTriple(lmodule.get());

        // TODO(lukas): Make configurable.
        llvm::errs() << *lmodule;
        llvm::errs().flush();
    }
} // namespace vast::hl

void vast::hl::registerHLToLLVMIR(mlir::DialectRegistry &registry)
{
    registry.insert< HighLevelDialect >();
    registry.addDialectInterface< HighLevelDialect, ToLLLVMIR >();
    registry.addDialectInterface< HighLevelDialect, ToLLLVMIR >();
}
void vast::hl::registerHLToLLVMIR(mlir::MLIRContext &ctx)
{
    mlir::DialectRegistry registry;
    registerHLToLLVMIR(registry);
    ctx.appendDialectRegistry(registry);
}

std::unique_ptr< mlir::Pass > vast::hl::createLLVMDumpPass()
{
    return std::make_unique< LLVMDump >();
}
