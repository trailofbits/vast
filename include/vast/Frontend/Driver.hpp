// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Driver/Compilation.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Driver/Driver.h>
#include <llvm/Support/BuryPointer.h>
#include <llvm/Support/CrashRecoveryContext.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Diagnostics.hpp"

namespace vast::cc {

    //
    // driver wrappers
    //
    using clang_driver      = clang::driver::Driver;
    using clang_compilation = clang::driver::Compilation;

    using repro_level   = clang_driver::ReproLevel;
    using driver_status = clang_driver::CommandStatus;
    using clang_command = clang::driver::Command;

    #ifdef _WIN32
        static constexpr bool win_32 = true;
    #else
        static constexpr bool win_32 = false;
    #endif

    #ifdef LLVM_ON_UNIX
        static constexpr bool llvm_on_linux = true;
    #else
        static constexpr bool llvm_on_linux = false;
    #endif

    struct driver {
        using exec_compile_t  = llvm::function_ref< int(const vast_args &, argv_storage &) >;
        using compilation_ptr = std::unique_ptr< clang_compilation >;

        driver(const std::string &path, const vast_args &vargs,
               argv_storage &cc_args, exec_compile_t cc1
        )
            : compile(cc1), vargs(vargs), cc_args(cc_args), diag(cc_args, path)
            , drv(path, llvm::sys::getDefaultTargetTriple(), diag.engine, "vast compiler")
        {
            // FIXME: use SetInstallDir(Args, TheDriver, CanonicalPrefixes);
            // FIXME: set target and mode

            // Ensure the CC1Command actually catches cc1 crashes
            llvm::CrashRecoveryContext::Enable();
        }

        compilation_ptr make_compilation() {
            return compilation_ptr( drv.BuildCompilation(cc_args) );
        }

        std::optional< repro_level > get_repro_level(const compilation_ptr &comp) const {
            std::optional< repro_level > level = repro_level::OnCrash;

            if (auto *arg = comp->getArgs().getLastArg(clang::driver::options::OPT_gen_reproducer_eq)) {
                level = llvm::StringSwitch< std::optional< repro_level > >(arg->getValue())
                    .Case("off", repro_level::Off)
                    .Case("crash", repro_level::OnCrash)
                    .Case("error", repro_level::OnError)
                    .Case("always", repro_level::Always)
                    .Default(std::nullopt);

                if (!level) {
                    llvm::errs() << "Unknown value for " << arg->getSpelling() << ": '"
                                 << arg->getValue() << "'\n";
                    return level;
                }
            }

            if (!!::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH")) {
                level = repro_level::Always;
            }

            return level;
        }

        const clang_command * first_job(const compilation_ptr &comp) {
            return &(*comp->getJobs().begin());
        }

        using failing_commands = llvm::SmallVector< std::pair< int, const clang_command * >, 4 >;

        int execute() {
            auto comp  = make_compilation();
            auto level = get_repro_level(comp);
            if (!level) {
                return 1;
            }

            int result = 1;
            bool is_crash = false;
            driver_status command_status = driver_status::Ok;

            const clang_command *failing_command = nullptr;
            if (!comp->getJobs().empty()) {
                failing_command = first_job(comp);
            }

            if (comp && !comp->containsError()) {
                failing_commands failing;
                result = drv.ExecuteCompilation(*comp, failing);

                for (const auto &[cmd_result, cmd] : failing) {
                    failing_command = cmd;
                    if (!result) {
                        result = cmd_result;
                    }

                    // If result status is < 0, then the driver command signalled an error.
                    // If result status is 70, then the driver command reported a fatal error.
                    // On Windows, abort will return an exit code of 3.  In these cases,
                    // generate additional diagnostic information if possible.
                    is_crash = cmd_result < 0 || cmd_result == 70;
                    if constexpr (win_32) {
                        is_crash |= cmd_result == 3;
                    }
                    // When running in integrated-cc1 mode, the CrashRecoveryContext returns
                    // the same codes as if the program crashed. See section "Exit Status for
                    // Commands":
                    // https://pubs.opengroup.org/onlinepubs/9699919799/xrat/V4_xcu_chap02.html
                    if constexpr (llvm_on_linux) {
                        is_crash |= cmd_result > 128;
                    }

                    command_status = is_crash ? driver_status::Crash : driver_status::Error;

                    if (is_crash) {
                        break;
                    }
                }
            }

            // Print the bug report message that would be printed if we did actually
            // crash, but only if we're crashing due to FORCE_CLANG_DIAGNOSTICS_CRASH.
            if (::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH"))
                llvm::dbgs() << llvm::getBugReportMsg();

            auto maybe_generate_compilation_diagnostics = [&] {
                return drv.maybeGenerateCompilationDiagnostics(command_status, *level, *comp, *failing_command);
            };

            if (failing_command != nullptr && maybe_generate_compilation_diagnostics()) {
                result = 1;
            }

            diag.finish();

            if (is_crash) {
                // When crashing in -fintegrated-cc1 mode, bury the timer pointers, because
                // the internal linked list might point to already released stack frames.
                llvm::BuryPointer(llvm::TimerGroup::aquireDefaultGroup());
            } else {
                // If any timers were active but haven't been destroyed yet, print their
                // results now.  This happens in -disable-free mode.
                llvm::TimerGroup::printAll(llvm::errs());
                llvm::TimerGroup::clearAll();
            }

            if constexpr (win_32) {
                // Exit status should not be negative on Win32, unless abnormal termination.
                // Once abnormal termination was caught, negative status should not be
                // propagated.
                if (result < 0) {
                    return 1;
                }
            }

            // If we have multiple failing commands, we return the result of the first
            // failing command.
            return result;
        }

        exec_compile_t compile;
        const vast_args &vargs;
        argv_storage & cc_args;

        errs_diagnostics diag;
        clang_driver drv;
    };

} // namespace vast::cc
