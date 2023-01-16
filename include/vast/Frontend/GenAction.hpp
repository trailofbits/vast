// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/FrontendAction.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Frontend/CompilerInstance.hpp"
#include "vast/Frontend/Generator.hpp"

namespace llvm {
    class LLVMIRContext;
    class Module;
} // namespace llvm

namespace mlir {
    class MLIRContext;
    class ModuleOp;
} // namespace mlir

namespace vast::cc {

    struct vast_gen_consumer;

    struct vast_gen_action : clang::ASTFrontendAction {
        enum class output_type {
            emit_assembly,
            emit_high_level,
            emit_cir,
            emit_llvm,
            emit_obj,
            none
        };

    private:
        friend struct vast_gen_consumer;

        OwningModuleRef mlir_module;
        // std::unique_ptr< llvm::Module > llvm_module;

        MContext *mcontext;

        OwningModuleRef load_module(llvm::MemoryBufferRef mref);

    protected:

        vast_gen_action(output_type action, MContext *mcontext = nullptr);

        std::unique_ptr< clang::ASTConsumer >
        CreateASTConsumer(compiler_instance &ci, llvm::StringRef InFile) override;

        void ExecuteAction() override;

        void EndSourceFileAction() override;

    public:
        virtual ~vast_gen_action() = default;

        vast_gen_consumer *consumer;
        output_type action;
    };

    //
    // Emit assembly
    //
    struct emit_assembly_action : vast_gen_action {
        emit_assembly_action(MContext *mcontext = nullptr);
    private:
        virtual void anchor();
    };

    //
    // Emit LLVM
    //
    struct emit_llvm_action : vast_gen_action {
        emit_llvm_action(MContext *mcontext = nullptr);
    private:
        virtual void anchor();
    };

    //
    // Emit obj
    //
    struct emit_obj_action : vast_gen_action {
        emit_obj_action(MContext *mcontext = nullptr);
    private:
        virtual void anchor();
    };

} // namespace vast
