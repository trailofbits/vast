// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Target/LLVMIR/Convert.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>

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

#include "vast/Dialect/Core/CoreOps.hpp"

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

    // TODO: move to translation passes that erase specific types from module
    void clean_up_data_layout(mlir_module mod) {
        // If the old data layout with high level types is left in the module,
        // some parsing functionality inside the `mlir::translateModuleToLLVMIR`
        // will fail and no conversion translation happens, even in case these
        // entries are not used at all.
        auto dl = mod.getDataLayoutSpec();

        auto is_llvm_compatible_entry = [] (auto entry) {
            return mlir::LLVM::isCompatibleType(entry.getKey().template get< mlir_type >());
        };

        auto filtered_entries = [&] {
            return llvm::to_vector(
                llvm::make_filter_range(dl.getEntries(),is_llvm_compatible_entry)
            );
        } ();

        mod->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName,
            mlir::DataLayoutSpecAttr::get(mod.getContext(), filtered_entries)
        );
    }

    std::unique_ptr< llvm::Module > translate(
        mlir_module mod, llvm::LLVMContext &llvm_ctx
    ) {
        clean_up_data_layout(mod);

        // TODO move to LLVM conversion and use attr replacer
        if (auto target = mod->getAttr(core::CoreDialect::getTargetTripleAttrName())) {
            auto triple = mlir::cast< mlir::StringAttr>(target);
            mod->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(), triple);
            mod->removeAttr(core::CoreDialect::getTargetTripleAttrName());
        }

        mlir::registerBuiltinDialectTranslation(*mod.getContext());
        mlir::registerLLVMDialectTranslation(*mod.getContext());

        return mlir::translateModuleToLLVMIR(mod, llvm_ctx);
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
