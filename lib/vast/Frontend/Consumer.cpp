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

#include "vast/CodeGen/Passes.hpp"
#include "vast/CodeGen/CodeGenContext.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"

#include "vast/Util/Common.hpp"

#include "vast/Frontend/Targets.hpp"
#include "vast/Target/LLVMIR/Convert.hpp"

namespace vast::cc {

    namespace llvmir = target::llvmir;

    using pipeline = llvmir::pipeline;

    [[nodiscard]] pipeline parse_pipeline(const std::optional< vast_args::option_list > &list);

    [[nodiscard]] pipeline parse_pipeline(string_ref from);

    [[nodiscard]] target_dialect parse_target_dialect(string_ref from);

    [[nodiscard]] std::string to_string(target_dialect target);

    void emit_mlir_output(target_dialect target, owning_module_ref mod, mcontext_t *mctx);

    using source_language = core::SourceLanguage;

    source_language get_source_language(const cc::language_options &opts);

    void vast_consumer::Initialize(acontext_t &actx) {
        VAST_CHECK(!mctx, "initialized multiple times");
        mctx = std::make_unique< mcontext_t >();
        cgctx = std::make_unique< cg::codegen_context >(
            *mctx, actx, get_source_language(opts.lang)
        );

        codegen = std::make_unique< cg::codegen_driver >(*cgctx, opts, vargs);
    }

    bool vast_consumer::HandleTopLevelDecl(clang::DeclGroupRef decls) {
        clang::PrettyStackTraceDecl crash_info(
            *decls.begin(), clang::SourceLocation(), cgctx->actx.getSourceManager(),
            "LLVM IR generation of declaration"
        );

        if (opts.diags.hasErrorOccurred()) {
            return true;
        }

        return codegen->handle_top_level_decl(decls), true;
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
        codegen->finalize();

        if (!vargs.has_option(opt::disable_vast_verifier)) {
            if (!codegen->verify_module()) {
                VAST_UNREACHABLE("codegen: module verification error before running vast passes");
            }
        }

        compile_via_vast(cgctx->mod.get(), mctx.get());
    }

    void vast_consumer::HandleTagDeclDefinition(clang::TagDecl *decl) {
        auto &actx = cgctx->actx;
        clang::PrettyStackTraceDecl crash_info(
            decl, clang::SourceLocation(), actx.getSourceManager(),
            "vast generation of declaration"
        );

        if (opts.diags.hasErrorOccurred()) {
            return;
        }

        // Don't allow re-entrant calls to generator triggered by PCH
        // deserialization to emit deferred decls.
        cg::defer_handle_of_top_level_decl handling_decl(
            *codegen, /* emit deferred */false
        );

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

    void vast_consumer::CompleteTentativeDefinition(clang::VarDecl *decl) {
        codegen->handle_top_level_decl(decl);
    }

    void vast_consumer::CompleteExternalDeclaration(clang::VarDecl * /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void vast_consumer::AssignInheritanceModel(clang::CXXRecordDecl * /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void vast_consumer::HandleVTable(clang::CXXRecordDecl * /* decl */) { VAST_UNIMPLEMENTED; }

    owning_module_ref vast_consumer::result() {
        return std::move(cgctx->mod);
    }

    //
    // vast stream consumer
    //

    void vast_stream_consumer::HandleTranslationUnit(acontext_t &actx) {
        base::HandleTranslationUnit(actx);
        auto mod = result();

        switch (action) {
            case output_type::emit_assembly:
                return emit_backend_output(
                    backend::Backend_EmitAssembly, std::move(mod), mctx.get()
                );
            case output_type::emit_mlir: {
                auto trg = parse_target_dialect(vargs.get_option(opt::emit_mlir).value());
                return emit_mlir_output(trg, std::move(mod), mctx.get());
            }
            case output_type::emit_llvm:
                return emit_backend_output(
                    backend::Backend_EmitLL, std::move(mod), mctx.get()
                );
            case output_type::emit_obj:
                return emit_backend_output(
                    backend::Backend_EmitObj, std::move(mod), mctx.get()
                );
            case output_type::none:
                break;
        }
    }

    void vast_stream_consumer::emit_backend_output(
        backend backend_action, owning_module_ref mlir_module, mcontext_t *mctx
    ) {
        llvm::LLVMContext llvm_context;
        llvmir::register_vast_to_llvm_ir(*mctx);
        auto pipeline = parse_pipeline(vargs.get_options_list(opt::opt_pipeline));
        llvmir::lower_hl_module(mlir_module.get(), pipeline);

        auto mod = llvmir::translate(mlir_module.get(), llvm_context);
        auto dl  = cgctx->actx.getTargetInfo().getDataLayoutString();
        clang::EmitBackendOutput(
            opts.diags, opts.headers, opts.codegen, opts.target, opts.lang, dl, mod.get(),
            backend_action, &opts.vfs, std::move(output_stream)
        );
    }

    void vast_stream_consumer::emit_mlir_output(
        target_dialect target, owning_module_ref mod, mcontext_t *mctx
    ) {
        if (!output_stream || !mod) {
            return;
        }

        auto setup_pipeline_and_execute = [&] {
            auto pipeline = parse_pipeline(vargs.get_options_list(opt::opt_pipeline));
            switch (target) {
                case target_dialect::high_level:
                    break;
                case target_dialect::llvm: {
                    // TODO: These should probably be moved outside of `target::llvmir`.
                    llvmir::register_vast_to_llvm_ir(*mctx);
                    llvmir::lower_hl_module(mod.get(), pipeline);
                    break;
                }
                // case target_dialect::cir: {
                // }
                default:
                    VAST_UNREACHABLE("Cannot emit {0}, missing support", to_string(target));
            }
        };

        // Handle source manager properly given that lifetime analysis
        // might emit warnings and remarks.
        auto &src_mgr     = cgctx->actx.getSourceManager();
        auto main_file_id = src_mgr.getMainFileID();

        auto file_buff = llvm::MemoryBuffer::getMemBuffer(
            src_mgr.getBufferOrFake(main_file_id)
        );

        llvm::SourceMgr mlir_src_mgr;
        mlir_src_mgr.AddNewSourceBuffer(std::move(file_buff), llvm::SMLoc());

        if (vargs.has_option(opt::vast_verify_diags)) {
            mlir::SourceMgrDiagnosticVerifierHandler src_mgr_handler(mlir_src_mgr, mctx);
            mctx->printOpOnDiagnostic(false);
            setup_pipeline_and_execute();

            // Verify the diagnostic handler to make sure that each of the
            // diagnostics matched.
            if (src_mgr_handler.verify().failed()) {
                llvm::sys::RunInterruptHandlers();
                VAST_UNREACHABLE("failed mlir codegen");
            }
        } else {
            mlir::SourceMgrDiagnosticHandler src_mgr_handler(mlir_src_mgr, mctx);
            setup_pipeline_and_execute();
        }

        // Emit remaining defaulted C++ methods
        // if (!vargs.has_option(opt::disable_emit_cxx_default)) {
        //     generator->build_default_methods();
        // }

        // FIXME: we cannot roundtrip prettyForm=true right now.
        mlir::OpPrintingFlags flags;
        flags.enableDebugInfo(vargs.has_option(opt::show_locs), /* prettyForm */ true);

        mod->print(*output_stream, flags);
    }

    void vast_consumer::compile_via_vast(vast_module mod, mcontext_t *mctx) {
        const bool enable_vast_verifier = !vargs.has_option(opt::disable_vast_verifier);
        auto pass = cg::emit_high_level_pass(mod, mctx, &cgctx->actx, enable_vast_verifier);
        if (pass.failed()) {
            VAST_UNREACHABLE("codegen: MLIR pass manager fails when running vast passes");
        }
    }

    source_language get_source_language(const cc::language_options &opts) {
        using ClangStd = clang::LangStandard;

        if (opts.CPlusPlus || opts.CPlusPlus11 || opts.CPlusPlus14 ||
            opts.CPlusPlus17 || opts.CPlusPlus20 || opts.CPlusPlus23 ||
            opts.CPlusPlus26)
            return source_language::CXX;
        if (opts.C99 || opts.C11 || opts.C17 || opts.C2x ||
            opts.LangStd == ClangStd::lang_c89)
            return source_language::C;

        // TODO: support remaining source languages.
        VAST_UNIMPLEMENTED_MSG("VAST does not yet support the given source language");
    }

    pipeline parse_pipeline(const std::optional< vast_args::option_list > &list) {
        if (!list) {
            return llvmir::default_pipeline();
        }

        if (list->size() != 1) {
            VAST_UNREACHABLE("Cannot use more than one pipeline!");
        }

        return parse_pipeline(list->front());
    }

    pipeline parse_pipeline(string_ref from) {
        auto trg = from.lower();
        if (trg == "with-abi") {
            return pipeline::with_abi;
        }
        if (trg == "baseline") {
            return pipeline::baseline;
        }

        VAST_UNREACHABLE("Unknown option of pipeline to use: {0}", trg);
    }


} // namespace vast::cc
