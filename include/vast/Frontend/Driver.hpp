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
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Diagnostics.hpp"
#include "vast/Frontend/Action.hpp"

#include "vast/Config/config.h"

namespace vast::cc {

    //
    // driver wrappers
    //
    using clang_driver      = clang::driver::Driver;
    using clang_compilation = clang::driver::Compilation;

    using parsed_clang_name = clang::driver::ParsedClangName;
    using toolchain         = clang::driver::ToolChain;

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

    static const char *get_stable_cstr(std::set<std::string> &saved_string, string_ref str) {
        return saved_string.insert(std::string(str)).first->c_str();
    }

    static void insert_target_and_mode_args(
        const parsed_clang_name &name_parts, argv_storage_base &cmd_args,
        std::set<std::string> &saved_string
    ) {
        // Put target and mode arguments at the start of argument list so that
        // arguments specified in command line could override them. Avoid putting
        // them at index 0, as an option like '-cc1' must remain the first.
        int insertion_point = 0;
        if (cmd_args.size() > 0)
            ++insertion_point;

        if (name_parts.DriverMode) {
            // Add the mode flag to the arguments.
            cmd_args.insert(cmd_args.begin() + insertion_point,
                            get_stable_cstr(saved_string, name_parts.DriverMode));
        }

        if (name_parts.TargetIsValid) {
            const char *arr[] = {"-target", get_stable_cstr(saved_string, name_parts.TargetPrefix)};
            cmd_args.insert(cmd_args.begin() + insertion_point, std::begin(arr), std::end(arr));
        }
    }

    template< typename result_type >
    static result_type check_env_var(const char *env_opt_set, const char *env_opt_file, std::string &opt_file) {
        if (const char *str = ::getenv(env_opt_set)) {
            result_type opt_val = str;
            if (const char *var = ::getenv(env_opt_file)) {
                opt_file = var;
            }
            return opt_val;
        }

        return result_type{};
    }

    //
    // vast::driver
    //
    struct driver {
        using exec_compile_t  = int (*)(argv_storage_base &);
        using compilation_ptr = std::unique_ptr< clang_compilation >;

        driver(const std::string &path, argv_storage_base &cmd_args, exec_compile_t cc1, bool canonical_prefixes)
            : cc1_entry_point(cc1), cmd_args(cmd_args), diag(cmd_args, path)
            , drv(path, llvm::sys::getDefaultTargetTriple(), diag.engine, "vast compiler")
        {
            set_install_dir(cmd_args, canonical_prefixes);

            auto target_and_mode = toolchain::getTargetAndModeFromProgramName(cmd_args[0]);
            drv.setTargetAndMode(target_and_mode);

            std::set<std::string> saved_string;
            // TODO fill saved strings

            insert_target_and_mode_args(target_and_mode, cmd_args, saved_string);

            preprocess_vast_args(cmd_args);
        }

        compilation_ptr make_compilation() {
            drv.CC1Main = cc1_entry_point;
            // Ensure the CC1Command actually catches cc1 crashes
            llvm::CrashRecoveryContext::Enable();

            return compilation_ptr( drv.BuildCompilation(cmd_args) );
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
            if (!set_backdoor_driver_outputs_from_env_vars()) {
                return 1;
            }

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

        bool set_backdoor_driver_outputs_from_env_vars() {
            drv.CCPrintOptions = check_env_var<bool>(
                "CC_PRINT_OPTIONS", "CC_PRINT_OPTIONS_FILE", drv.CCPrintOptionsFilename
            );

            // TODO:
            // if (check_env_var<bool>("CC_PRINT_HEADERS", "CC_PRINT_HEADERS_FILE", drv.CCPrintHeadersFilename)) {
            //     drv.CCPrintHeadersFormat = HIFMT_Textual;
            //     drv.CCPrintHeadersfiltering = HIFIL_None;
            // } else {
            //     auto env_var = check_env_var<std::string>(
            //         "CC_PRINT_HEADERS_FORMAT", "CC_PRINT_HEADERS_FILE",
            //         drv.CCPrintHeadersFilename
            //     );

            //     if (!env_var.empty()) {
            //         drv.CCPrintHeadersFormat = stringToHeaderIncludeFormatKind(env_var.c_str());
            //         if (!drv.CCPrintHeadersFormat) {
            //             drv.Diag(clang::diag::err_drv_print_header_env_var) << 0 << env_var;
            //             return false;
            //         }

            //         const char *filtering_string = ::getenv("CC_PRINT_HEADERS_FILTERING");
            //         HeaderIncludefilteringKind filtering;

            //         if (!stringToHeaderIncludefiltering(filtering_string, filtering)) {
            //             drv.Diag(clang::diag::err_drv_print_header_env_var) << 1 << filtering_string;
            //             return false;
            //         }

            //         if ((drv.CCPrintHeadersFormat == HIFMT_Textual && filtering != HIFIL_None) ||
            //             (drv.CCPrintHeadersFormat == HIFMT_JSON && filtering != HIFIL_Only_Direct_System)
            //         ) {
            //             drv.Diag(clang::diag::err_drv_print_header_env_var_combination) << env_var << filtering_string;
            //             return false;
            //         }
            //         drv.CCPrintHeadersfiltering = filtering;
            //     }
            // }

            drv.CCLogDiagnostics = check_env_var<bool>(
                "CC_LOG_DIAGNOSTICS", "CC_LOG_DIAGNOSTICS_FILE",
                drv.CCLogDiagnosticsFilename
            );

            drv.CCPrintProcessStats = check_env_var<bool>(
                "CC_PRINT_PROC_STAT", "CC_PRINT_PROC_STAT_FILE",
                drv.CCPrintStatReportFilename
            );

            return true;
        }

        void preprocess_vast_args(argv_storage_base &all_args) {
            auto [vargs, ccargs] = vast::cc::filter_args(all_args);
            // force no link step in case of emiting mlir file
            if (vast::cc::opt::emit_only_mlir(vargs)) {
                all_args.push_back("-c");
            }

            if (vast::cc::opt::emit_only_llvm(vargs)) {
                all_args.push_back("-emit-llvm");
            }

            auto is_resource_dir = [](llvm::StringRef arg) {
                return arg.starts_with("-resource-dir");
            };

            if (std::ranges::count_if(ccargs, is_resource_dir) == 0) {
                all_args.push_back("-resource-dir");
                auto &res_arg = cached_strings.emplace_back(
                    clang_driver::GetResourcesPath(CLANG_BINARY_PATH, "")
                );
                all_args.push_back(res_arg.data());
            }
        }

        void set_install_dir(argv_storage_base &argv, bool canonical_prefixes) {
            // Attempt to find the original path used to invoke the driver, to determine
            // the installed path. We do this manually, because we want to support that
            // path being a symlink.
            llvm::SmallString< 128 > installed_path(argv[0]);

            // Do a PATH lookup, if there are no directory components.
            if (llvm::sys::path::filename(installed_path) == installed_path) {
                if (auto tmp = llvm::sys::findProgramByName(llvm::sys::path::filename(installed_path.str()))) {
                    installed_path = *tmp;
                }
            }

            // FIXME: We don't actually canonicalize this, we just make it absolute.
            if (canonical_prefixes) {
                llvm::sys::fs::make_absolute(installed_path);
            }

            string_ref installed_path_parent(llvm::sys::path::parent_path(installed_path));
            if (llvm::sys::fs::exists(installed_path_parent)) {
                drv.setInstalledDir(installed_path_parent);
            }
        }


        exec_compile_t cc1_entry_point;
        argv_storage_base &cmd_args;
        std::vector< std::string > cached_strings;

        errs_diagnostics diag;
        clang_driver drv;
    };

} // namespace vast::cc
