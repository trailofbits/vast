// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/GenAction.hpp"
#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/CodeGen/BackendUtil.h>
#include <llvm/Support/Signals.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Frontend/Options.hpp"
#include "vast/Frontend/Common.hpp"
#include "vast/Frontend/Diagnostics.hpp"
#include "vast/CodeGen/Passes.hpp"

namespace clang {
    class CXXRecordDecl;
    class DeclGroupRef;
    class FunctionDecl;
    class TagDecl;
    class VarDecl;
} // namespace clang

namespace vast::cc {

    namespace opt {

        bool emit_only_mlir(const vast_args &vargs) {
            for (auto arg : {emit_high_level, emit_cir}) {
                if (vargs.has_option(arg))
                    return true;
            }

            return false;
        }

    } // namespace opt

    using output_stream_ptr = std::unique_ptr< llvm::raw_pwrite_stream >;

    static std::string get_output_stream_suffix(output_type act) {
        switch (act) {
            case output_type::emit_assembly: return "s";
            case output_type::emit_high_level: return "hl";
            case output_type::emit_cir: return "cir";
            case output_type::emit_llvm: return "ll";
            case output_type::emit_obj: return "o";
            case output_type::none: break;
        }

        throw compiler_error("unsupported action type");
    }

    static auto get_output_stream(compiler_instance &ci, string_ref in, output_type act)
        -> output_stream_ptr
    {
        if (act == output_type::none) {
            return nullptr;
        }

        return ci.createDefaultOutputFile(false, in, get_output_stream_suffix(act));
    }

    struct vast_gen_consumer : cg::clang_ast_consumer {

        using vast_generator_ptr = std::unique_ptr< cg::vast_generator >;

        vast_gen_consumer(
            output_type act,
            diagnostics_engine &diags,
            header_search_options &hopts,
            codegen_options &copts,
            target_options &topts,
            language_options &lopts,
            // frontend_options &fopts,
            const vast_args &vargs,
            output_stream_ptr os
        )
            : action(act)
            , diags(diags)
            , header_search_opts(hopts)
            , codegen_opts(copts)
            , target_opts(topts)
            , lang_opts(lopts)
            // , frontend_opts(fopts)
            , vargs(vargs)
            , output_stream(std::move(os))
            , generator(std::make_unique< cg::vast_generator >(diags, codegen_opts))
        {}

        void Initialize(acontext_t &ctx) override {
            assert(!acontext && "initialized multiple times");
            acontext = &ctx;
            generator->Initialize(ctx);
        }

        bool HandleTopLevelDecl(clang::DeclGroupRef decls) override {
            clang::PrettyStackTraceDecl crash_info(
                *decls.begin(), clang::SourceLocation(), acontext->getSourceManager(),
                "LLVM IR generation of declaration"
            );
            return generator->HandleTopLevelDecl(decls);
        }

        void HandleCXXStaticMemberVarInstantiation(clang::VarDecl * /* decl */) override {
            throw compiler_error("HandleCXXStaticMemberVarInstantiation not implemented");
        }

        void HandleInlineFunctionDefinition(clang::FunctionDecl * /* decl */) override {
            throw compiler_error("HandleInlineFunctionDefinition not implemented");
        }

        void HandleInterestingDecl(clang::DeclGroupRef /* decl */) override {
            throw compiler_error("HandleInterestingDecl not implemented");
        }

        void emit_backend_output(clang::BackendAction backend_action) {
            // llvm::LLVMcontext_t llvm_context;
            throw compiler_error("HandleTranslationUnit for emit llvm not implemented");

            std::unique_ptr< llvm::Module > mod = nullptr /* todo lower_from_vast_to_llvm */;
            clang::EmitBackendOutput(
                  diags
                , header_search_opts
                , codegen_opts
                , target_opts
                , lang_opts
                , acontext->getTargetInfo().getDataLayoutString()
                , mod.get()
                , backend_action
                , std::move(output_stream)
            );
        }

        enum class target_dialect {
            high_level
        };

        void emit_mlir_output(target_dialect target, owning_module_ref mod, mcontext_t *mctx) {
            if (!output_stream || !mod) {
                return;
            }

            const bool disable_vast_verifier = vargs.has_option(opt::disable_vast_verifier);

            auto execute_vast_pipeline = [&] {
                // FIXME: parse pass options and deal with different passes in more sane way
                switch (target) {
                    case target_dialect::high_level: {
                        return cg::emit_high_level_pass(mod.get(), mctx, acontext, disable_vast_verifier);
                    }
                }

                throw cc::compiler_error("codegen: unsupported target dialect");
            };

            auto setup_vast_pipeline_and_execute = [&] {
                if (execute_vast_pipeline().failed()) {
                    throw cc::compiler_error("codegen: MLIR pass manager fails when running vast passes");
                }
            };

            // Handle source manager properly given that lifetime analysis
            // might emit warnings and remarks.
            auto &src_mgr = acontext->getSourceManager();
            auto main_file_id = src_mgr.getMainFileID();

            auto file_buff = llvm::MemoryBuffer::getMemBuffer(
                src_mgr.getBufferOrFake(main_file_id)
            );

            llvm::SourceMgr mlir_src_mgr;
            mlir_src_mgr.AddNewSourceBuffer(std::move(file_buff), llvm::SMLoc());

            if (vargs.has_option(opt::vast_verify_diags)) {
                mlir::SourceMgrDiagnosticVerifierHandler src_mgr_handler(mlir_src_mgr, mctx);
                mctx->printOpOnDiagnostic(false);
                setup_vast_pipeline_and_execute();

                // Verify the diagnostic handler to make sure that each of the
                // diagnostics matched.
                if (src_mgr_handler.verify().failed()) {
                    llvm::sys::RunInterruptHandlers();
                    throw cc::compiler_error("failed mlir codegen");
                }
            } else {
                mlir::SourceMgrDiagnosticHandler src_mgr_handler(mlir_src_mgr, mctx);
                setup_vast_pipeline_and_execute();
            }

            // Emit remaining defaulted C++ methods
            // if (!vargs.has_option(opt::disable_emit_cxx_default)) {
            //     generator->build_default_methods();
            // }

            // FIXME: we cannot roundtrip prettyForm=true right now.
            mlir::OpPrintingFlags flags;
            flags.enableDebugInfo(/* prettyForm */ false);
            mod->print(*output_stream, flags);
        }

        void HandleTranslationUnit(acontext_t &acontext) override {
            // Note that this method is called after `HandleTopLevelDecl` has already
            // ran all over the top level decls. Here clang mostly wraps defered and
            // global codegen, followed by running vast passes.
            generator->HandleTranslationUnit(acontext);

            if (!vargs.has_option(opt::disable_vast_verifier)) {
                if (!generator->verify_module()) {
                    throw compiler_error("codegen: module verification error before running vast passes");
                }
            }

            auto mod  = generator->freeze();
            auto mctx = generator->take_context();

            switch (action) {
                case output_type::emit_assembly:
                    return emit_backend_output(clang::BackendAction::Backend_EmitAssembly);
                case output_type::emit_high_level:
                    return emit_mlir_output(target_dialect::high_level, std::move(mod), mctx.get());
                case output_type::emit_cir:
                    throw compiler_error("HandleTranslationUnit for emit CIR not implemented");
                case output_type::emit_llvm:
                    return emit_backend_output(clang::BackendAction::Backend_EmitLL);
                case output_type::emit_obj:
                    return emit_backend_output(clang::BackendAction::Backend_EmitObj);
                case output_type::none: break;
            }
        }

        void HandleTagDeclDefinition(clang::TagDecl *decl) override {
            clang::PrettyStackTraceDecl crash_info(
                decl, clang::SourceLocation(), acontext->getSourceManager(),
                "vast generation of declaration"
            );

            generator->HandleTagDeclDefinition(decl);
        }

        // void HandleTagDeclRequiredDefinition(clang::TagDecl */* decl */) override {
        //     throw compiler_error("HandleTagDeclRequiredDefinition not implemented");
        // }

        void CompleteTentativeDefinition(clang::VarDecl */* decl */) override {
            throw compiler_error("CompleteTentativeDefinition not implemented");
        }

        void CompleteExternalDeclaration(clang::VarDecl */* decl */) override {
            throw compiler_error("CompleteExternalDeclaration not implemented");
        }

        void AssignInheritanceModel(clang::CXXRecordDecl */* decl */) override {
            throw compiler_error("AssignInheritanceModel not implemented");
        }

        void HandleVTable(clang::CXXRecordDecl */* decl */) override {
            throw compiler_error("HandleVTable not implemented");
        }

      private:
        virtual void anchor() {}

        output_type action;

        diagnostics_engine &diags;

        // options
        const header_search_options &header_search_opts;
        const codegen_options &codegen_opts;
        const target_options &target_opts;
        const language_options &lang_opts;
        // const frontend_options &frontend_opts;

        const vast_args &vargs;

        output_stream_ptr output_stream;

        acontext_t *acontext = nullptr;

        vast_generator_ptr generator;
    };

    vast_gen_action::vast_gen_action(output_type act, const vast_args &vargs, mcontext_t *montext)
        : action(act), mcontext(montext ? montext : new mcontext_t), vargs(vargs)
    {}

    owning_module_ref vast_gen_action::load_module(llvm::MemoryBufferRef /* mref */) {
        throw compiler_error("load_module not implemented");
    }

    void vast_gen_action::ExecuteAction() {
        // FIXME: if (getCurrentFileKind().getLanguage() != Language::CIR)
        this->ASTFrontendAction::ExecuteAction();
    }

    auto vast_gen_action::CreateASTConsumer(compiler_instance &ci, string_ref input)
        -> std::unique_ptr< clang::ASTConsumer >
    {
        auto out = ci.takeOutputStream();
        if (!out) {
            out = get_output_stream(ci, input, action);
        }

        auto result = std::make_unique< vast_gen_consumer >(
              action
            , ci.getDiagnostics()
            , ci.getHeaderSearchOpts()
            , ci.getCodeGenOpts()
            , ci.getTargetOpts()
            , ci.getLangOpts()
            // , ci.getFrontendOpts()
            , vargs
            , std::move(out)
        );

        consumer = result.get();

        // Enable generating macro debug info only when debug info is not disabled and
        // also macrod ebug info is enabled
        auto &cgo = ci.getCodeGenOpts();
        auto nodebuginfo = clang::codegenoptions::NoDebugInfo;
        if (cgo.getDebugInfo() != nodebuginfo && cgo.MacroDebugInfo) {
            throw compiler_error("Macro debug info not implemented");
        }

        return result;
    }

    void vast_gen_action::EndSourceFileAction() {
        // If the consumer creation failed, do nothing.
        if (!getCompilerInstance().hasASTConsumer())
            return;

        // TODO: pass the module around
    }

    void emit_assembly_action::anchor() {}

    emit_assembly_action::emit_assembly_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_assembly, vargs, mcontex)
    {}

    void emit_llvm_action::anchor() {}

    emit_llvm_action::emit_llvm_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_llvm, vargs, mcontex)
    {}

    void emit_obj_action::anchor() {}

    emit_obj_action::emit_obj_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_obj, vargs, mcontex)
    {}

    void emit_high_level_action::anchor() {}

    emit_high_level_action::emit_high_level_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_high_level, vargs, mcontex)
    {}

    void emit_cir_action::anchor() {}

    emit_cir_action::emit_cir_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_cir, vargs, mcontex)
    {}

} // namespace vast::cc
