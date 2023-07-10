// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Target/LLVMIR/Convert.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <mlir/Pass/PassManager.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>

#include <llvm/ADT/TypeSwitch.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Conversion/Passes.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"

namespace vast::target::llvmir
{
    class ToLLVMIR : public mlir::LLVMTranslationDialectInterface
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

    std::unique_ptr< llvm::Module > translate(mlir::Operation *op,
                                              llvm::LLVMContext &llvm_ctx,
                                              const std::string &module_name)
    {
        auto mlir_module = mlir::dyn_cast< mlir::ModuleOp >(op);
        VAST_CHECK(mlir_module, "Cannot translate operation that is not an mlir::ModuleOp!");

        auto mctx = mlir_module->getContext();

        // If the old data layout with high level types is left in the module,
        // some parsing functionality inside the `mlir::translateModuleToLLVMIR`
        // will fail and no conversion translation happens, even in case these
        // entries are not used at all.
        auto old_dl = op->getAttr(mlir::DLTIDialect::kDataLayoutAttrName);
        mlir_module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                             mlir::DataLayoutSpecAttr::get(mctx, {}));

        auto lmodule = mlir::translateModuleToLLVMIR(mlir_module, llvm_ctx);

        // Restore the data layout in case this module is getting re-used later.
        mlir_module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, old_dl);

        // Setup proper triplet.
        mlir::ExecutionEngine::setupTargetTriple(lmodule.get());

        return lmodule;
    }

    void prepare_hl_module(mlir::Operation *op)
    {
        auto mctx = op->getContext();
        mlir::PassManager pm(mctx);

        // TODO(target:llvmir): This should be refactored out as a pipeline so we
        //                      can run it from command line as well.
        // TODO(target:llvmir): Add missing passes.
        pm.addPass(hl::createHLLowerTypesPass());
        pm.addPass(hl::createDCEPass());
        pm.addPass(hl::createResolveTypeDefsPass());
        pm.addPass(createHLToLLCFPass());
        pm.addPass(createHLToLLVarsPass());
        pm.addPass(createHLEmitLazyRegionsPass());
        pm.addPass(createHLToLLGEPsPass());
        pm.addPass(createHLStructsToLLVMPass());
        pm.addPass(createIRsToLLVMPass());
        pm.addPass(createCoreToLLVMPass());

        pm.enableIRPrinting([](auto *, auto *) { return false; }, // before
                            [](auto *, auto *) { return true; }, //after
                            false, // module scope
                            false, // after change
                            true, // after failure
                            llvm::errs());

        auto run_result = pm.run(op);

        VAST_CHECK(mlir::succeeded(run_result), "Some pass in prepare_module() failed");
    }

    void register_vast_to_llvm_ir(mlir::DialectRegistry &registry)
    {
        registry.insert< hl::HighLevelDialect >();
        mlir::registerAllToLLVMIRTranslations(registry);
    }

    void register_vast_to_llvm_ir(mcontext_t &mctx)
    {
        mlir::DialectRegistry registry;
        register_vast_to_llvm_ir(registry);
        mctx.appendDialectRegistry(registry);
    }

} // namespace vast::target::llvmir
