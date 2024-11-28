// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Consumer.hpp"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

VAST_RELAX_WARNINGS
#include <llvm/Support/Signals.h>

#include <mlir/Bytecode/BytecodeWriter.h>

#include <mlir/Pass/PassManager.h>

#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
VAST_UNRELAX_WARNINGS

#include <filesystem>
#include <fstream>

#include "vast/CodeGen/CodeGenDriver.hpp"

#include "vast/Util/Common.hpp"

#include "vast/Frontend/Pipelines.hpp"
#include "vast/Frontend/Sarif.hpp"
#include "vast/Frontend/Targets.hpp"

#include "vast/Target/LLVMIR/Convert.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Config/config.h"

namespace vast::cc {

    [[nodiscard]] target_dialect parse_target_dialect(string_ref from);

    [[nodiscard]] std::string to_string(target_dialect target);

    void vast_consumer::Initialize(acontext_t &actx) {
        VAST_CHECK(!driver, "initialized multiple times");
        driver = cg::mk_default_driver(opts, vargs, actx, mctx);
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
        driver->finalize();
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

    void vast_consumer::AssignInheritanceModel(clang::CXXRecordDecl * /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void vast_consumer::HandleVTable(clang::CXXRecordDecl * /* decl */) { VAST_UNIMPLEMENTED; }

    owning_mlir_module_ref vast_consumer::result() { return driver->freeze(); }

    //
    // vast stream consumer
    //

    std::optional< target_dialect > get_target_dialect(const vast_args &vargs) {
        if (vargs.has_option(opt::emit_mlir_after)) {
            // Pretend to emit all the way to LLVM and
            // let the pass scheduler decide where to stop.
            return target_dialect::llvm;
        }

        if (auto trg = vargs.get_option(opt::emit_mlir)) {
            return parse_target_dialect(trg.value());
        }

        return std::nullopt;
    }

    void vast_stream_consumer::HandleTranslationUnit(acontext_t &actx) {
        base::HandleTranslationUnit(actx);
        auto mod = result();

        switch (action) {
            case output_type::emit_assembly:
                return emit_backend_output(backend::Backend_EmitAssembly, std::move(mod));
            case output_type::emit_mlir: {
                if (auto trg = get_target_dialect(vargs)) {
                    return emit_mlir_output(trg.value(), std::move(mod));
                } else {
                    VAST_FATAL("no target dialect specified for MLIR output");
                }
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
        backend backend_action, owning_mlir_module_ref mod
    ) {
        llvm::LLVMContext llvm_context;
        process_mlir_module(target_dialect::llvm, mod.get());

        auto final_mlir_module = mlir::cast< mlir_module >(mod->getBody()->front());
        auto llvm_mod          = target::llvmir::translate(final_mlir_module, llvm_context);
        auto dl                = driver->acontext().getTargetInfo().getDataLayoutString();

        clang::EmitBackendOutput(
            opts.diags, opts.headers, opts.codegen, opts.target, opts.lang, dl, llvm_mod.get(),
            backend_action, &opts.vfs, std::move(output_stream)
        );
    }

    namespace sarif {
        struct diagnostics;
    } // namespace sarif

    #ifdef VAST_ENABLE_SARIF
    std::unique_ptr< vast::cc::sarif::diagnostics > setup_sarif_diagnostics(
        const vast_args &vargs, mcontext_t &mctx
    ) {
        if (vargs.get_option(opt::output_sarif)) {
                auto diags = std::make_unique< vast::cc::sarif::diagnostics >(vargs);
                mctx.getDiagEngine().registerHandler(diags->handler());
                return diags;
        } else {
            return nullptr;
        }
    }
    #endif // VAST_ENABLE_SARIF

    void emit_sarif_diagnostics(
        vast::cc::sarif::diagnostics &&sarif_diagnostics, logical_result result, string_ref path
    ) {
        #ifdef VAST_ENABLE_SARIF
            std::error_code ec;
            llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
            if (ec) {
                VAST_FATAL("Failed to open file for SARIF output: {}", ec.message());
            }

            nlohmann::json report = std::move(sarif_diagnostics).emit(result);
            os << report.dump(2);
        #else
            VAST_REPORT("SARIF support is disabled");
        #endif // VAST_ENABLE_SARIF
    }

    void vast_stream_consumer::process_mlir_module(target_dialect target, mlir_module mod) {
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

        #ifdef VAST_ENABLE_SARIF
        auto sarif_diagnostics = setup_sarif_diagnostics(vargs, mctx);
        #endif // VAST_ENABLE_SARIF

        auto result = pipeline->run(mod);

        VAST_CHECK(
            mlir::succeeded(result), "MLIR pass manager failed when running vast passes"
        );

        // Verify the diagnostic handler to make sure that each of the
        // diagnostics matched.
        if (verify_diagnostics && src_mgr_handler.verify().failed()) {
            llvm::sys::RunInterruptHandlers();
            VAST_FATAL("failed mlir codegen");
        }

        if (auto path = vargs.get_option(opt::output_sarif)) {
            #ifdef VAST_ENABLE_SARIF
            if (sarif_diagnostics) {
                emit_sarif_diagnostics(
                    std::move(*sarif_diagnostics), result, path.value().str()
                );
            } else {
                VAST_REPORT("SARIF diagnostics are missing");
            }
            #else
                VAST_REPORT("SARIF support is disabled");
            #endif // VAST_ENABLE_SARIF
        }

        // Emit remaining defaulted C++ methods
        // if (!vargs.has_option(opt::disable_emit_cxx_default)) {
        //     generator->build_default_methods();
        // }
    }

    void vast_stream_consumer::emit_mlir_output(
        target_dialect target, owning_mlir_module_ref mod
    ) {
        if (!output_stream || !mod) {
            return;
        }

        process_mlir_module(target, mod.get());

        if (vargs.has_option(opt::emit_mlir_bytecode)) {
            print_mlir_bytecode(std::move(mod));
        } else {
            print_mlir_string_format(std::move(mod));
        }
    }

    void vast_stream_consumer::print_mlir_bytecode(owning_mlir_module_ref mod) {
        mlir::BytecodeWriterConfig config("VAST");
        if (mlir::failed(mlir::writeBytecodeToFile(mod.get(), *output_stream, config))) {
            VAST_FATAL("Could not generate mlir bytecode");
        }
    }

    void vast_stream_consumer::print_mlir_string_format(owning_mlir_module_ref mod) {
        // FIXME: we cannot roundtrip prettyForm=true right now.
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(
            vargs.has_option(opt::show_locs), /* prettyForm */ !vargs.has_option(opt::loc_attrs)
        );

        mod->print(*output_stream, flags);
    }

} // namespace vast::cc
