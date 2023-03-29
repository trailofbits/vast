//
// Copyright (c) 2022, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.
//

//===----------------------------------------------------------------------===//
//
// This is the entry point to the vast-front -cc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Basic/TargetOptions.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Support/Timer.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/CompilerInstance.hpp"
#include "vast/Frontend/CompilerInvocation.hpp"
#include "vast/Frontend/Diagnostics.hpp"

using namespace vast::cc;

static void error_handler(void *user_data, const char *msg, bool get_crash_diag) {
  auto &diags = *static_cast< clang::DiagnosticsEngine* >(user_data);

  diags.Report(clang::diag::err_fe_error_backend) << msg;

  // Run the interrupt handlers to make sure any special cleanups get done, in
  // particular that we remove files registered with RemoveFileOnSignal.
  llvm::sys::RunInterruptHandlers();

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error.  Otherwise, exit with status 1.
  llvm::sys::Process::Exit(get_crash_diag ? 70 : 1);
}

namespace vast::cc {

    bool execute_compiler_invocation(compiler_instance *ci, const vast_args &vargs);

    int cc1(const vast_args &vargs, argv_t ccargs, arg_t tool, void *main_addr) {
        // FIXME: ensureSufficientStack

        auto comp = std::make_unique< compiler_instance >();
        // FIXME: register the support for object-file-wrapped Clang modules.

        // Initialize targets first, so that --version shows registered targets.
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
        llvm::InitializeAllAsmParsers();

        vast::cc::buffered_diagnostics diags(ccargs);

        // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
        // if (find(argv, StringRef("-Rround-trip-cc1-args")) != argv.end()) {
        //     diags.setSeverity(diag::remark_cc1_round_trip_generated, diag::Severity::Remark, {});
        // }

        auto success = compiler_invocation::create_from_args(comp->getInvocation(), diags.engine, ccargs, tool);

        auto &frontend_opts = comp->getFrontendOpts();
        // auto &target_opts   = comp->getFrontendOpts();
        auto &header_opts   = comp->getHeaderSearchOpts();

        if (frontend_opts.TimeTrace || !frontend_opts.TimeTracePath.empty()) {
            frontend_opts.TimeTrace = 1;
            llvm::timeTraceProfilerInitialize(frontend_opts.TimeTraceGranularity, tool);
        }

        // --print-supported-cpus takes priority over the actual compilation.
        // if (opts.PrintSupportedCPUs) {
        //     return PrintSupportedCPUs(target_opts.Triple);
        // }

        // Infer the builtin include path if unspecified.
        if (header_opts.UseBuiltinIncludes && header_opts.ResourceDir.empty()) {
            header_opts.ResourceDir = clang_invocation::GetResourcesPath(tool, main_addr);
        }

        // Create the actual diagnostics engine.
        if (comp->createDiagnostics(); !comp->hasDiagnostics()) {
            return 1;
        }

        // Set an error handler, so that any LLVM backend diagnostics go through our
        // error handler.
        llvm::install_fatal_error_handler(error_handler, static_cast<void*>(&comp->getDiagnostics()));

        diags.flush();
        if (!success) {
            comp->getDiagnosticClient().finish();
            return 1;
        }

        // Execute the frontend actions.
        try {
            llvm::TimeTraceScope TimeScope("ExecuteCompiler");
            success = execute_compiler_invocation(comp.get(), vargs);
        } catch ( ... ) {
            // TODO( vast-front ): This is required as `~clang::CompilerInstance` would
            //                     fire an assert as stack unwinds.
            comp->setSema(nullptr);
            comp->setASTConsumer(nullptr);
            comp->clearOutputFiles(true);
            throw;
        }

        // If any timers were active but haven't been destroyed yet, print their
        // results now.  This happens in -disable-free mode.
        llvm::TimerGroup::printAll(llvm::errs());
        llvm::TimerGroup::clearAll();

        using small_string = llvm::SmallString<128>;

        if (llvm::timeTraceProfilerEnabled()) {
            small_string path(frontend_opts.OutputFile);
            llvm::sys::path::replace_extension(path, "json");

            if (!frontend_opts.TimeTracePath.empty()) {
                // replace the suffix to '.json' directly
                small_string trace_path(frontend_opts.TimeTracePath);
                if (llvm::sys::fs::is_directory(trace_path)) {
                    llvm::sys::path::append(trace_path, llvm::sys::path::filename(path));
                }

                path.assign(trace_path);
            }

            if (auto profiler_output = comp->createOutputFile(
                    path.str(), /*Binary=*/false, /*RemoveFileOnSignal=*/false,
                    /*useTemporary=*/false)) {
                llvm::timeTraceProfilerWrite(*profiler_output);
                profiler_output.reset();
                llvm::timeTraceProfilerCleanup();
                comp->clearOutputFiles(false);
            }
        }

        // Our error handler depends on the Diagnostics object, which we're
        // potentially about to delete. Uninstall the handler now so that any
        // later errors use the default handling behavior instead.
        llvm::remove_fatal_error_handler();

        // When running with -disable-free, don't do any destruction or shutdown.
        if (frontend_opts.DisableFree) {
            llvm::BuryPointer(std::move(comp));
            return !success;
        }

        return !success;
    }

} // namespace vast::cc
