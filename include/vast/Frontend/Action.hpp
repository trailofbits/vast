// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/FrontendAction.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Frontend/CompilerInstance.hpp"
#include "vast/Frontend/FrontendAction.hpp"
#include "vast/Frontend/Options.hpp"

namespace llvm {
    class LLVMIRContext;
    class Module;
} // namespace llvm

namespace mlir {
    class MLIRContext;
    class ModuleOp;
} // namespace mlir

namespace vast::cc {

    struct vast_consumer;

    struct vast_action : frontend_action {
        virtual ~vast_action() = default;

        vast_consumer *consumer;
        output_type action;

    protected:

        vast_action(output_type action, const vast_args &vargs);

        std::unique_ptr< clang::ASTConsumer >
        CreateASTConsumer(compiler_instance &ci, string_ref input) override;

        void ExecuteAction() override;

        void EndSourceFileAction() override;

    private:
        friend struct vast_consumer;

        owning_module_ref mlir_module;

        std::unique_ptr< mcontext_t > mcontext = std::make_unique< mcontext_t >();

        owning_module_ref load_module(llvm::MemoryBufferRef mref);

        const vast_args &vargs;
    };

    //
    // Emit assembly
    //
    struct emit_assembly_action : vast_action {
        explicit emit_assembly_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit LLVM
    //
    struct emit_llvm_action : vast_action {
        explicit emit_llvm_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit MLIR
    //
    struct emit_mlir_action : vast_action {
        explicit emit_mlir_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit obj
    //
    struct emit_obj_action : vast_action {
        explicit emit_obj_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

} // namespace vast
