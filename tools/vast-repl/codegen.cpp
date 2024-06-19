// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/codegen.hpp"
#include "vast/repl/common.hpp"

#include "vast/repl/state.hpp"

VAST_RELAX_WARNINGS
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <llvm/Support/Signals.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Action.hpp"
#include "vast/Frontend/CompilerInstance.hpp"
#include "vast/Frontend/CompilerInvocation.hpp"
#include "vast/Frontend/Diagnostics.hpp"

#include <fstream>

namespace vast::cc {
    bool execute_compiler_invocation(cc::compiler_instance *ci, const cc::vast_args &vargs);
} // namespace vast::cc

namespace vast::repl::codegen {

    std::unique_ptr< clang::ASTUnit > ast_from_source(string_ref source) {
        return clang::tooling::buildASTFromCode(source);
    }

    static void error_handler(void *user_data, const char *msg, bool get_crash_diag) {
        auto &diags = *static_cast< clang::DiagnosticsEngine* >(user_data);

        diags.Report(clang::diag::err_fe_error_backend) << msg;

        // Run the interrupt handlers to make sure any special cleanups get done, in
        // particular that we remove files registered with RemoveFileOnSignal.
        llvm::sys::RunInterruptHandlers();
    }

    owning_module_ref emit_module(const std::filesystem::path &source, mcontext_t */* mctx */) {
        // TODO setup args from repl state
        std::vector< const char * > ccargs = { source.c_str() };
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

        auto mctx = cc::mk_mcontext();
        if (auto action = std::make_unique< vast::cc::emit_mlir_module >(vargs, *mctx); !action) {
            comp->ExecuteAction(*action);
            llvm::remove_fatal_error_handler();
            return action->result();
        }

        return {};
    }

} // namespace vast::repl::codegen
