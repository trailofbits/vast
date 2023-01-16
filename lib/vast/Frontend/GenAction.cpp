// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/GenAction.hpp"

VAST_RELAX_WARNINGS
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/BackendUtil.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Common.hpp"
#include "vast/Frontend/Diagnostics.hpp"


namespace clang {
    class CXXRecordDecl;
    class DeclGroupRef;
    class FunctionDecl;
    class TagDecl;
    class VarDecl;
} // namespace clang

namespace vast::cc {

    struct vast_gen_consumer : clang_ast_consumer {

        using output_stream_t = std::unique_ptr< llvm::raw_pwrite_stream >;

        using vast_generator_ptr = std::unique_ptr< vast_generator >;

        vast_gen_consumer(
            vast_gen_action::output_type act,
            diagnostics_engine &diags,
            header_search_options &hopts,
            codegen_options &copts,
            target_options &topts,
            language_options &lopts,
            // frontend_options &fopts,
            output_stream_t os
        )
            : action(act)
            , diags(diags)
            , header_search_opts(hopts)
            , codegen_opts(copts)
            , target_opts(topts)
            , lang_opts(lopts)
            // , frontend_opts(fopts)
            , output_stream(std::move(os))
            , generator(std::make_unique< vast_generator >(diags, codegen_opts))
        {}

        void Initialize(AContext &ctx) override {
            assert(!acontext && "initialized multiple times");
            acontext = &ctx;
            generator->Initialize(ctx);
        }

        bool HandleTopLevelDecl(clang::DeclGroupRef /* decl */) override {
            throw compiler_error("HandleTopLevelDecl not implemented");
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
            // llvm::LLVMContext llvm_context;
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

        void HandleTranslationUnit(AContext &acontext) override {
            using output_type = vast_gen_action::output_type;

            // Note that this method is called after `HandleTopLevelDecl` has already
            // ran all over the top level decls. Here clang mostly wraps defered and
            // global codegen, followed by running CIR passes.
            generator->HandleTranslationUnit(acontext);

            switch (action) {
                case output_type::emit_assembly:
                    return emit_backend_output(clang::BackendAction::Backend_EmitAssembly);
                case output_type::emit_high_level:
                    throw compiler_error("HandleTranslationUnit for emit HL not implemented");
                case output_type::emit_cir:
                    throw compiler_error("HandleTranslationUnit for emit CIR not implemented");
                case output_type::emit_llvm:
                    return emit_backend_output(clang::BackendAction::Backend_EmitLL);
                case output_type::emit_obj:
                    return emit_backend_output(clang::BackendAction::Backend_EmitObj);
                case output_type::none: break;
            }
        }

        void HandleTagDeclDefinition(clang::TagDecl */* decl */) override {
            throw compiler_error("HandleTagDeclDefinition not implemented");
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

        vast_gen_action::output_type action;

        diagnostics_engine &diags;

        // options
        const header_search_options &header_search_opts;
        const codegen_options &codegen_opts;
        const target_options &target_opts;
        const language_options &lang_opts;
        // const frontend_options &frontend_opts;

        output_stream_t output_stream;

        AContext *acontext = nullptr;

        vast_generator_ptr generator;
    };

    vast_gen_action::vast_gen_action(output_type act, MContext *montext)
        : mcontext(montext ? montext : new MContext), action(act)
    {}

    OwningModuleRef vast_gen_action::load_module(llvm::MemoryBufferRef /* mref */) {
        throw compiler_error("load_module not implemented");
    }

    void vast_gen_action::ExecuteAction() {
        throw compiler_error("ExecuteAction not implemented");
    }

    auto vast_gen_action::CreateASTConsumer(compiler_instance &ci, llvm::StringRef /* input */)
        -> std::unique_ptr< clang::ASTConsumer >
    {
        auto out = ci.takeOutputStream();
        if (!out) {
            throw compiler_error("getOutputStream not implemented");
            //     out = getOutputStream(ci, inputFile, action);
        }

        auto result = std::make_unique< vast_gen_consumer >(
              action
            , ci.getDiagnostics()
            , ci.getHeaderSearchOpts()
            , ci.getCodeGenOpts()
            , ci.getTargetOpts()
            , ci.getLangOpts()
            // , ci.getFrontendOpts()
            , std::move(out)
        );

        consumer = result.get();

        // Enable generating macro debug info only when debug info is not disabled and
        // also macrod ebug info is enabled
        // if (ci.getCodeGenOpts().getDebugInfo() != codegenoptions::NoDebugInfo &&
        //     ci.getCodeGenOpts().MacroDebugInfo) {
        //     llvm_unreachable("NYI");
        //     throw compiler_error("CreateASTConsumer not implemented");
        // }

        return result;
    }

    void vast_gen_action::EndSourceFileAction() {
        throw compiler_error("EndSourceFileAction not implemented");
    }

    void emit_assembly_action::anchor() {}

    emit_assembly_action::emit_assembly_action(MContext *mcontex)
        : vast_gen_action(output_type::emit_assembly, mcontex)
    {}

    void emit_llvm_action::anchor() {}

    emit_llvm_action::emit_llvm_action(MContext *mcontex)
        : vast_gen_action(output_type::emit_llvm, mcontex)
    {}

    void emit_obj_action::anchor() {}

    emit_obj_action::emit_obj_action(MContext *mcontex)
        : vast_gen_action(output_type::emit_obj, mcontex)
    {}

} // namespace vast::cc
