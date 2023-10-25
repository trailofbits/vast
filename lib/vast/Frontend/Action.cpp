// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Action.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Frontend/Consumer.hpp"

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
            for (auto arg : { emit_mlir }) {
                if (vargs.has_option(arg)) {
                    return true;
                }
            }

            return false;
        }

        bool emit_only_llvm(const vast_args &vargs) { return vargs.has_option(emit_llvm); }

    } // namespace opt

    static std::string get_output_stream_suffix(output_type act) {
        switch (act) {
            case output_type::emit_assembly:
                return "s";
            case output_type::emit_mlir:
                return "mlir";
            case output_type::emit_llvm:
                return "ll";
            case output_type::emit_obj:
                return "o";
            case output_type::none:
                break;
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

    vast_action::vast_action(output_type act, const vast_args &vargs)
        : action(act), vargs(vargs)
    {}

    void vast_action::ExecuteAction() {
        // FIXME: if (getCurrentFileKind().getLanguage() != Language::CIR)
        this->ASTFrontendAction::ExecuteAction();
    }

    auto vast_action::CreateASTConsumer(compiler_instance &ci, string_ref input)
        -> std::unique_ptr< clang::ASTConsumer >
    {
        auto out = ci.takeOutputStream();
        if (!out) {
            out = get_output_stream(ci, input, action);
        }

        auto result = std::make_unique< vast_consumer >(
            action, options(ci), vargs, std::move(out)
        );

        consumer = result.get();

        // Enable generating macro debug info only when debug info is not disabled and
        // also macrod ebug info is enabled
        auto &cgo        = ci.getCodeGenOpts();
        auto nodebuginfo = llvm::codegenoptions::NoDebugInfo;
        if (cgo.getDebugInfo() != nodebuginfo && cgo.MacroDebugInfo) {
            VAST_UNIMPLEMENTED_MSG("Macro debug info not implemented");
        }

        return result;
    }

    void vast_action::EndSourceFileAction() {
        // If the consumer creation failed, do nothing.
        if (!getCompilerInstance().hasASTConsumer()) {
            return;
        }

        // TODO: pass the module around
    }

    // emit assembly
    void emit_assembly_action::anchor() {}

    emit_assembly_action::emit_assembly_action(const vast_args &vargs)
        : vast_action(output_type::emit_assembly, vargs)
    {}

    // emit_llvm
    void emit_llvm_action::anchor() {}

    emit_llvm_action::emit_llvm_action(const vast_args &vargs)
        : vast_action(output_type::emit_llvm, vargs)
    {}

    // emit_mlir
    void emit_mlir_action::anchor() {}

    emit_mlir_action::emit_mlir_action(const vast_args &vargs)
        : vast_action(output_type::emit_mlir, vargs)
    {}

    // emit_obj
    void emit_obj_action::anchor() {}

    emit_obj_action::emit_obj_action(const vast_args &vargs)
        : vast_action(output_type::emit_obj, vargs)
    {}

} // namespace vast::cc
