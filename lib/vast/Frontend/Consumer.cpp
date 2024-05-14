// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Consumer.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/Signals.h>

#include <mlir/Pass/PassManager.h>

#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
VAST_UNRELAX_WARNINGS

#include <filesystem>

#include "vast/CodeGen/CodeGenDriver.hpp"

#include "vast/Util/Common.hpp"

#include "vast/Frontend/Pipelines.hpp"
#include "vast/Frontend/Targets.hpp"

#include "vast/Target/LLVMIR/Convert.hpp"

namespace vast::cc {

    [[nodiscard]] target_dialect parse_target_dialect(string_ref from);

    [[nodiscard]] std::string to_string(target_dialect target);

    void vast_consumer::Initialize(acontext_t &actx) {
        VAST_CHECK(!driver, "initialized multiple times");
        driver = cg::mk_driver(opts, vargs, actx);
    }

    bool vast_consumer::HandleTopLevelDecl(clang::DeclGroupRef decls) {
        if (opts.diags.hasErrorOccurred()) {
            return true;
        }

        driver->emit(decls);
        return true;
    }

    void vast_consumer::HandleCXXStaticMemberVarInstantiation(clang::VarDecl * /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void vast_consumer::HandleInlineFunctionDefinition(clang::FunctionDecl * /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void vast_consumer::HandleInterestingDecl(clang::DeclGroupRef /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void vast_consumer::HandleTranslationUnit(acontext_t &actx) {
        // Note that this method is called after `HandleTopLevelDecl` has already
        // ran all over the top level decls. Here clang mostly wraps defered and
        // global codegen, followed by running vast passes.
        driver->finalize(vargs);
    }

    void vast_consumer::HandleTagDeclDefinition(clang::TagDecl *decl) {
        if (opts.diags.hasErrorOccurred()) {
            return;
        }

        auto &actx = decl->getASTContext();

        // For MSVC compatibility, treat declarations of static data members with
        // inline initializers as definitions.
        if (actx.getTargetInfo().getCXXABI().isMicrosoft()) {
            VAST_UNIMPLEMENTED;
        }

        // For OpenMP emit declare reduction functions, if required.
        if (actx.getLangOpts().OpenMP) {
            VAST_UNIMPLEMENTED;
        }
    }

    // void vast_consumer::HandleTagDeclRequiredDefinition(clang::TagDecl */* decl */) {
    //     VAST_UNIMPLEMENTED;
    // }

    void vast_consumer::CompleteTentativeDefinition(clang::VarDecl * /* decl */) {}

    void vast_consumer::CompleteExternalDeclaration(clang::VarDecl * /* decl */) {}

    void vast_consumer::AssignInheritanceModel(clang::CXXRecordDecl * /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void vast_consumer::HandleVTable(clang::CXXRecordDecl * /* decl */) { VAST_UNIMPLEMENTED; }

    owning_module_ref vast_consumer::result() {
        return driver->freeze();
    }

    //
    // vast stream consumer
    //

    void vast_stream_consumer::HandleTranslationUnit(acontext_t &actx) {
        base::HandleTranslationUnit(actx);
        auto mod = result();

        switch (action) {
            case output_type::emit_assembly:
                return emit_backend_output(backend::Backend_EmitAssembly, std::move(mod));
            case output_type::emit_mlir: {
                if (auto trg = vargs.get_option(opt::emit_mlir)) {
                    return emit_mlir_output(parse_target_dialect(trg.value()), std::move(mod));
                }
                VAST_FATAL("no target dialect specified for MLIR output");
            }
            case output_type::emit_llvm:
                return emit_backend_output(backend::Backend_EmitLL, std::move(mod));
            case output_type::emit_obj:
                return emit_backend_output(backend::Backend_EmitObj, std::move(mod));
            case output_type::none:
                break;
        }
    }

    void vast_stream_consumer::emit_backend_output(
        backend backend_action, owning_module_ref mod
    ) {
        llvm::LLVMContext llvm_context;

        process_mlir_module(target_dialect::llvm, mod.get());

        auto llvm_mod = target::llvmir::translate(mod.get(), llvm_context);
        auto dl  = driver->acontext().getTargetInfo().getDataLayoutString();

        clang::EmitBackendOutput(
            opts.diags, opts.headers, opts.codegen, opts.target, opts.lang, dl, llvm_mod.get(),
            backend_action, &opts.vfs, std::move(output_stream)
        );
    }

    void vast_stream_consumer::process_mlir_module(
        target_dialect target, mlir::ModuleOp mod
    ) {
        // Handle source manager properly given that lifetime analysis
        // might emit warnings and remarks.
        auto &src_mgr     = driver->acontext().getSourceManager();
        auto &mctx        = driver->mcontext();
        auto main_file_id = src_mgr.getMainFileID();

        auto file_buff = llvm::MemoryBuffer::getMemBuffer(
            src_mgr.getBufferOrFake(main_file_id)
        );

        llvm::SourceMgr mlir_src_mgr;
        mlir_src_mgr.AddNewSourceBuffer(std::move(file_buff), llvm::SMLoc());

        bool verify_diagnostics = vargs.has_option(opt::vast_verify_diags);

        mlir::SourceMgrDiagnosticVerifierHandler src_mgr_handler(mlir_src_mgr, &mctx);

        if (vargs.has_option(opt::debug)) {
            mctx.printOpOnDiagnostic(true);
            mctx.printStackTraceOnDiagnostic(true);
            llvm::DebugFlag = true;
        }

        // Setup and execute vast pipeline
        auto file_entry = src_mgr.getFileEntryRefForID(main_file_id);
        VAST_CHECK(file_entry, "failed to recover file entry ref");
        auto snapshot_prefix = std::filesystem::path(file_entry->getName().str()).stem().string();

        auto pipeline = setup_pipeline(pipeline_source::ast, target, mctx, vargs, snapshot_prefix);
        VAST_CHECK(pipeline, "failed to setup pipeline");

        auto result = pipeline->run(mod);
        VAST_CHECK(mlir::succeeded(result), "MLIR pass manager failed when running vast passes");

        // Verify the diagnostic handler to make sure that each of the
        // diagnostics matched.
        if (verify_diagnostics && src_mgr_handler.verify().failed()) {
            llvm::sys::RunInterruptHandlers();
            VAST_FATAL("failed mlir codegen");
        }

        // Emit remaining defaulted C++ methods
        // if (!vargs.has_option(opt::disable_emit_cxx_default)) {
        //     generator->build_default_methods();
        // }
    }

    void vast_stream_consumer::emit_mlir_output(
        target_dialect target, owning_module_ref mod
    ) {
        if (!output_stream || !mod) {
            return;
        }

        process_mlir_module(target, mod.get());

        // FIXME: we cannot roundtrip prettyForm=true right now.
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(vargs.has_option(opt::show_locs), /* prettyForm */ true);

        mod->print(*output_stream, flags);
    }

} // namespace vast::cc
