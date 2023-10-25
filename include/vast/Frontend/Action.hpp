// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/FrontendAction.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Frontend/FrontendAction.hpp"
#include "vast/Frontend/CompilerInstance.hpp"
#include "vast/CodeGen/Generator.hpp"

namespace llvm {
    class LLVMIRContext;
    class Module;
} // namespace llvm

namespace mlir {
    class MLIRContext;
    class ModuleOp;
} // namespace mlir

namespace vast::cc {

    namespace opt {
        constexpr string_ref emit_llvm = "emit-llvm";
        constexpr string_ref emit_obj  = "emit-obj";
        constexpr string_ref emit_asm  = "emit-asm";

        constexpr string_ref emit_mlir = "emit-mlir";

        constexpr string_ref emit_locs = "emit-locs";

        constexpr string_ref opt_pipeline  = "pipeline";

        constexpr string_ref disable_vast_verifier = "disable-vast-verifier";
        constexpr string_ref vast_verify_diags = "verify-diags";
        constexpr string_ref disable_emit_cxx_default = "disable-emit-cxx-default";

        bool emit_only_mlir(const vast_args &vargs);
        bool emit_only_llvm(const vast_args &vargs);

    } // namespace opt

    struct vast_gen_consumer;

    struct vast_gen_action : frontend_action {
        virtual ~vast_gen_action() = default;

        vast_gen_consumer *consumer;
        output_type action;

    protected:

        vast_gen_action(output_type action, const vast_args &vargs);

        std::unique_ptr< clang::ASTConsumer >
        CreateASTConsumer(compiler_instance &ci, string_ref input) override;

        void ExecuteAction() override;

        void EndSourceFileAction() override;

    private:
        friend struct vast_gen_consumer;

        owning_module_ref mlir_module;

        std::unique_ptr< mcontext_t > mcontext = std::make_unique< mcontext_t >();

        owning_module_ref load_module(llvm::MemoryBufferRef mref);

        const vast_args &vargs;
    };

    //
    // Emit assembly
    //
    struct emit_assembly_action : vast_gen_action {
        explicit emit_assembly_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit LLVM
    //
    struct emit_llvm_action : vast_gen_action {
        explicit emit_llvm_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit MLIR
    //
    struct emit_mlir_action : vast_gen_action {
        explicit emit_mlir_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

    //
    // Emit obj
    //
    struct emit_obj_action : vast_gen_action {
        explicit emit_obj_action(const vast_args &vargs);
    private:
        virtual void anchor();
    };

} // namespace vast
