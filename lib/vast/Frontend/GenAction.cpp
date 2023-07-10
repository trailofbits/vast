// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/GenAction.hpp"
#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/CodeGen/BackendUtil.h>

#include <llvm/Support/Signals.h>

#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Frontend/Options.hpp"
#include "vast/Frontend/Diagnostics.hpp"
#include "vast/CodeGen/Passes.hpp"

#include "vast/Target/LLVMIR/Convert.hpp"

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
            for (auto arg : {emit_high_level, emit_cir, emit_mlir}) {
                if (vargs.has_option(arg))
                    return true;
            }

            return false;
        }

        bool emit_only_llvm(const vast_args &vargs) {
            return vargs.has_option(emit_llvm);
        }

    } // namespace opt

    using output_stream_ptr = std::unique_ptr< llvm::raw_pwrite_stream >;

    static std::string get_output_stream_suffix(output_type act) {
        switch (act) {
            case output_type::emit_assembly: return "s";
            case output_type::emit_high_level: return "hl";
            case output_type::emit_cir: return "cir";
            case output_type::emit_mlir: return "mlir";
            case output_type::emit_llvm: return "ll";
            case output_type::emit_obj: return "o";
            case output_type::none: break;
        }

        VAST_UNREACHABLE("unsupported action type");
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
            VAST_CHECK(!acontext, "initialized multiple times");
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
            VAST_UNIMPLEMENTED;
        }

        void HandleInlineFunctionDefinition(clang::FunctionDecl * /* decl */) override {
            VAST_UNIMPLEMENTED;
        }

        void HandleInterestingDecl(clang::DeclGroupRef /* decl */) override {
            VAST_UNIMPLEMENTED;
        }
        void emit_backend_output(clang::BackendAction backend_action,
                                 owning_module_ref mlir_module, mcontext_t *mctx)
        {
            llvm::LLVMContext llvm_context;
            target::llvmir::register_vast_to_llvm_ir(*mctx);
            target::llvmir::prepare_hl_module(mlir_module.get());
            auto mod = target::llvmir::translate(mlir_module.get(), llvm_context, "tmp");

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

        // TODO: Introduce helper wrapper on top of `vast_args`?
        enum class target_dialect {
            high_level
            , low_level
            , llvm
        };

        target_dialect parse_target_dialect(const vast_args::maybe_option_list &list)
        {
            if (!list)
                return target_dialect::high_level;

            if (list->size() != 1)
                VAST_UNREACHABLE("Can emit only one dialect.");

            return parse_target_dialect(list->front());
        }

        target_dialect parse_target_dialect(llvm::StringRef from)
        {
            auto trg = from.lower();
            if (trg == "hl" || trg == "high_level")
                return target_dialect::high_level;
            if (trg == "ll" || trg == "low_level")
                return target_dialect::low_level;
            if (trg == "llvm")
                return target_dialect::llvm;
            VAST_UNREACHABLE("Unknown option of target dialect: {0}", trg);
        }

        [[nodiscard]] static inline std::string to_string(target_dialect target)
        {
            switch (target)
            {
                case target_dialect::high_level: return "high_level";
                case target_dialect::low_level: return "low_level";
                case target_dialect::llvm: return "llvm";
            }
        }

        void compile_via_vast(auto mod, mcontext_t *mctx)
        {
            const bool disable_vast_verifier = vargs.has_option(opt::disable_vast_verifier);
            auto pass = cg::emit_high_level_pass(mod, mctx,
                                                 acontext, disable_vast_verifier);
            if (pass.failed())
                VAST_UNREACHABLE("codegen: MLIR pass manager fails when running vast passes");

        }

        void emit_mlir_output(target_dialect target, owning_module_ref mod, mcontext_t *mctx) {
            if (!output_stream || !mod) {
                return;
            }

            auto setup_pipeline_and_execute = [&]
            {
                switch (target)
                {
                    case target_dialect::high_level:
                        break;
                    case target_dialect::llvm:
                    {
                        // TODO: These should probably be moved outside of `target::llvmir`.
                        target::llvmir::register_vast_to_llvm_ir(*mctx);
                        target::llvmir::prepare_hl_module(mod.get());
                        break;
                    }
                    default:
                        VAST_UNREACHABLE("Cannot emit {0}, missing support", to_string(target));
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
                    VAST_UNREACHABLE("codegen: module verification error before running vast passes");
                }
            }

            auto mod  = generator->freeze();
            auto mctx = generator->take_context();

            compile_via_vast(mod.get(), mctx.get());

            switch (action)
            {
                case output_type::emit_assembly:
                    return emit_backend_output(clang::BackendAction::Backend_EmitAssembly,
                                               std::move(mod), mctx.get());
                case output_type::emit_high_level:
                    return emit_mlir_output(target_dialect::high_level, std::move(mod), mctx.get());
                case output_type::emit_cir:
                    VAST_UNIMPLEMENTED_MSG("HandleTranslationUnit for emit CIR not implemented");
                case output_type::emit_mlir:
                {
                    auto trg = parse_target_dialect(vargs.get_options_list(opt::emit_mlir));
                    return emit_mlir_output(trg, std::move(mod), mctx.get());
                }
                case output_type::emit_llvm:
                    return emit_backend_output(clang::BackendAction::Backend_EmitLL,
                                               std::move(mod), mctx.get());
                case output_type::emit_obj:
                    return emit_backend_output(clang::BackendAction::Backend_EmitObj,
                                               std::move(mod), mctx.get());
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
        //     VAST_UNIMPLEMENTED;
        // }

        void CompleteTentativeDefinition(clang::VarDecl *decl) override {
            generator->CompleteTentativeDefinition(decl);
        }

        void CompleteExternalDeclaration(clang::VarDecl */* decl */) override {
            VAST_UNIMPLEMENTED;
        }

        void AssignInheritanceModel(clang::CXXRecordDecl */* decl */) override {
            VAST_UNIMPLEMENTED;
        }

        void HandleVTable(clang::CXXRecordDecl */* decl */) override {
            VAST_UNIMPLEMENTED;
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
        VAST_UNIMPLEMENTED;
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
            VAST_UNIMPLEMENTED_MSG("Macro debug info not implemented");
        }

        return result;
    }

    void vast_gen_action::EndSourceFileAction() {
        // If the consumer creation failed, do nothing.
        if (!getCompilerInstance().hasASTConsumer())
            return;

        // TODO: pass the module around
    }

    // emit assembly
    void emit_assembly_action::anchor() {}

    emit_assembly_action::emit_assembly_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_assembly, vargs, mcontex)
    {}

    // emit_llvm
    void emit_llvm_action::anchor() {}

    emit_llvm_action::emit_llvm_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_llvm, vargs, mcontex)
    {}

    // emit_mlir
    void emit_mlir_action::anchor() {}

    emit_mlir_action::emit_mlir_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_mlir, vargs, mcontex)
    {}

    // emit_obj
    void emit_obj_action::anchor() {}

    emit_obj_action::emit_obj_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_obj, vargs, mcontex)
    {}

    // emit high level
    void emit_high_level_action::anchor() {}

    emit_high_level_action::emit_high_level_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_high_level, vargs, mcontex)
    {}

    // emit cir
    void emit_cir_action::anchor() {}

    emit_cir_action::emit_cir_action(const vast_args &vargs, mcontext_t *mcontex)
        : vast_gen_action(output_type::emit_cir, vargs, mcontex)
    {}

} // namespace vast::cc
