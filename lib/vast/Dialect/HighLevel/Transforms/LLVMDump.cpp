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

        // If the old data layout with high level types is left in the module,
        // some parsing functionality inside the `mlir::translateModuleToLLVMIR`
        // will fail and no conversion translation happens, even in case these
        // entries are not used at all.
        auto old_dl = op->getAttr(mlir::DLTIDialect::kDataLayoutAttrName);
        op->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                    mlir::DataLayoutSpecAttr::get(&mctx, {}));

        llvm::LLVMContext lctx;
        auto lmodule = mlir::translateModuleToLLVMIR(op, lctx);
        if (!lmodule)
            return signalPassFailure();

        // Restore the data layout in case this module is getting re-used later.
        op->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, old_dl);

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        mlir::ExecutionEngine::setupTargetTriple(lmodule.get());

        auto dump = [&](auto &stream)
        {
            stream << *lmodule;
            stream.flush();
        };

        auto outname = this->bitcode_file.getValue();
        if (outname.empty())
            return dump(llvm::outs());

        std::error_code ec;
        llvm::raw_fd_ostream out(outname, ec);

        VAST_CHECK(!ec, "Cannot store bitcode: {0}", ec.message());
        dump(out);
    }
} // namespace vast::hl

void vast::hl::registerHLToLLVMIR(mlir::DialectRegistry &registry)
{
    registry.insert< HighLevelDialect >();
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
