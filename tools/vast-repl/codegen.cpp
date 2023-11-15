// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/codegen.hpp"
#include "vast/repl/common.hpp"

#include "vast/repl/state.hpp"

#include <llvm/Support/Signals.h>

#include "vast/CodeGen/CodeGen.hpp"
#include "vast/Frontend/Action.hpp"
#include "vast/Frontend/CompilerInstance.hpp"
#include "vast/Frontend/CompilerInvocation.hpp"
#include "vast/Frontend/Diagnostics.hpp"

#include <fstream>

namespace vast::cc {
    bool execute_compiler_invocation(cc::compiler_instance *ci, const cc::vast_args &vargs);
} // namespace vast::cc

namespace vast::repl::codegen {

    std::string slurp(std::ifstream& in) {
        std::ostringstream sstr;
        sstr << in.rdbuf();
        return sstr.str();
    }

    std::unique_ptr< clang::ASTUnit > ast_from_source(const std::string &source) {
        return clang::tooling::buildASTFromCode(source);
    }

    std::string get_source(std::filesystem::path source) {
        std::ifstream in(source);
        return slurp(in);
    }

    static void error_handler(void *user_data, const char *msg, bool get_crash_diag) {
        auto &diags = *static_cast< clang::DiagnosticsEngine* >(user_data);

        diags.Report(clang::diag::err_fe_error_backend) << msg;

        // Run the interrupt handlers to make sure any special cleanups get done, in
        // particular that we remove files registered with RemoveFileOnSignal.
        llvm::sys::RunInterruptHandlers();
    }

    owning_module_ref emit_module(const std::string &/* source */, mcontext_t */* mctx */) {
        // TODO setup args from repl state
        const char *ccargs = {""};
        vast::cc::buffered_diagnostics diags(ccargs);

        auto comp = std::make_unique< cc::compiler_instance >();

        auto success = cc::compiler_invocation::create_from_args(
            comp->getInvocation(), diags.engine, ccargs, "vast-repl"
        );

        // Create the actual diagnostics engine.
        if (comp->createDiagnostics(); !comp->hasDiagnostics()) {
            return {};
        }

        // Set an error handler, so that any LLVM backend diagnostics go through our
        // error handler.
        llvm::install_fatal_error_handler(
            error_handler, static_cast<void*>(&comp->getDiagnostics())
        );

        diags.flush();
        if (!success) {
            comp->getDiagnosticClient().finish();
            return {};
        }

        // TODO setup args from repl state
        cc::vast_args vargs = {};

        comp->LoadRequestedPlugins();

        // If there were errors in processing arguments, don't do anything else.
        if (comp->getDiagnostics().hasErrorOccurred()) {
            return {};
        }

        if (auto action = std::make_unique< vast::cc::emit_mlir_module >(vargs); !action) {
            comp->ExecuteAction(*action);
            llvm::remove_fatal_error_handler();
            return action->result();
        }

        return {};
    }

} // namespace vast::repl::codegen
