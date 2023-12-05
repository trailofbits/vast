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
#include "vast/Frontend/Targets.hpp"

namespace llvm {
    class LLVMIRContext;
    class Module;
} // namespace llvm

namespace mlir {
    class MLIRContext;
    class ModuleOp;
} // namespace mlir

namespace vast::cc {

    struct vast_stream_consumer;

    //
    // Stream action produces the desired output
    // into output stream, created as part of ast consumer
    //
    struct vast_stream_action : frontend_action {
        virtual ~vast_stream_action() = default;

        vast_stream_consumer *consumer;
        output_type action;

    protected:

        vast_stream_action(output_type action, const vast_args &vargs);

        std::unique_ptr< clang::ASTConsumer >
        CreateASTConsumer(compiler_instance &ci, string_ref input) override;

        void ExecuteAction() override;

        void EndSourceFileAction() override;

    private:
        friend struct vast_consumer;

        const vast_args &vargs;
    };

    struct vast_consumer;

    //
    // Module action create MLIR module without dumping it to any output stream.
    // Users can retrieve it by calling `result` method.
    //
    struct vast_module_action : frontend_action {
        virtual ~vast_module_action() = default;

        owning_module_ref result();

        vast_consumer *consumer;
    protected:

        explicit vast_module_action(const vast_args &vargs);

        std::unique_ptr< clang::ASTConsumer >
        CreateASTConsumer(compiler_instance &ci, string_ref input) override;

        void ExecuteAction() override;

        void EndSourceFileAction() override;

    private:
        friend struct vast_consumer;

        const vast_args &vargs;
    };

    //
    // Emit assembly
    //
    struct emit_assembly_action : vast_stream_action {
        explicit emit_assembly_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit LLVM
    //
    struct emit_llvm_action : vast_stream_action {
        explicit emit_llvm_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit MLIR
    //
    struct emit_mlir_action : vast_stream_action {
        explicit emit_mlir_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit obj
    //
    struct emit_obj_action : vast_stream_action {
        explicit emit_obj_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit MLIR Module
    //
    struct emit_mlir_module : vast_module_action {
        explicit emit_mlir_module(const vast_args &vargs);
    private:
        virtual void anchor();
    };

} // namespace vast
