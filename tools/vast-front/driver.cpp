//
// Copyright (c) 2022, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.
//

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/VirtualFileSystem.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Driver.hpp"
#include "vast/Frontend/Options.hpp"

// main frontend method. Lives inside cc1_main.cpp
namespace vast::cc {
    extern int cc1(const vast_args & vargs, argv_t argv, arg_t tool, void *main_addr);
} // namespace vast::cc

VAST_RELAX_WARNINGS
std::string get_executable_path(vast::cc::arg_t tool, bool canonical_prefixes) {
    if (!canonical_prefixes) {
        llvm::SmallString<128> executable_path(tool);
        // Do a PATH lookup if Argv0 isn't a valid path.
        if (!llvm::sys::fs::exists(executable_path)) {
            if (auto prog = llvm::sys::findProgramByName(executable_path)) {
                executable_path = *prog;
            }
        }
        return std::string(executable_path.str());
    }

    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void *p = (void *) (intptr_t) get_executable_path;
    return llvm::sys::fs::getMainExecutable(tool, p);
}
VAST_UNRELAX_WARNINGS


static int execute_cc1_tool(vast::cc::argv_storage_base &cmd_args) {
    // If we call the cc1 tool from the clangDriver library (through
    // Driver::CC1Main), we need to clean up the options usage count. The options
    // are currently global, and they might have been used previously by the
    // driver.
    llvm::cl::ResetAllOptionOccurrences();

    llvm::BumpPtrAllocator pointer_allocator;
    llvm::StringSaver saver(pointer_allocator);
    llvm::cl::ExpandResponseFiles(
        saver, &llvm::cl::TokenizeGNUCommandLine, cmd_args
    );

    llvm::StringRef tool = cmd_args[1];

    VAST_RELAX_WARNINGS
    void *get_executable_path_ptr = (void *) (intptr_t) get_executable_path;
    VAST_UNRELAX_WARNINGS

    auto [vargs, ccargs] = vast::cc::filter_args(cmd_args);

    if (tool == "-cc1") {
        auto ccargs_ref = llvm::ArrayRef(ccargs).slice(2);
        return vast::cc::cc1(vargs, ccargs_ref, cmd_args[0], get_executable_path_ptr);
    }

    llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
                 << "Valid tools include '-cc1'.\n";
    return 1;
}

bool has_canonical_prefixes_option(const vast::cc::argv_storage &args) {
    bool result = true;

    for (auto arg : args) {
        // Skip end-of-line response file markers
        if (arg == nullptr)
            continue;
        if (vast::string_ref(arg) == "-canonical-prefixes") {
            result = true;
        } else if (vast::string_ref(arg) == "-no-canonical-prefixes") {
            result = false;
        }
    }

    return result;
}

void preprocess_vast_arguments(vast::cc::argv_storage &args) {
    auto plugin_arg = "-Xclang";
    // annotate vast arguments as plugin arguments to not be rejected as unknown arguments
    auto is_plugin_argument = [&] (auto it) {
        return vast::string_ref( *std::prev(it) ) == plugin_arg;
    };

    auto make_plugin_argument = [&] (auto it) {
        if (is_plugin_argument(it))
            return it;
        return std::next(args.insert(it, plugin_arg));
    };

    for (auto it = args.begin(); it != args.end(); it++) {
        auto arg = vast::string_ref(*it);
        if (arg.startswith(vast::cc::vast_option_prefix)) {
            it = make_plugin_argument(it);
        }
    }
}

int main(int argc, char **argv) try {
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);

    auto msg = llvm::formatv(
        "PLEASE submit a bug report to {0} and include the crash backtrace, "
        "preprocessed source, and associated run script.\n", vast::bug_report_url
    ).str();

    llvm::setBugReportMsg(msg.c_str());

    vast::cc::argv_storage cmd_args(argv, argv + argc);

    if (llvm::sys::Process::FixupStandardFileDescriptors()) {
        return 1;
    }

    llvm::InitializeAllTargets();

    llvm::BumpPtrAllocator pointer_allocator;
    llvm::StringSaver saver(pointer_allocator);

    // FIXME: deal with CL mode

    // Check if vast-front is in the frontend mode
    auto first_arg = llvm::find_if(llvm::drop_begin(cmd_args), [] (auto a) { return a != nullptr; });
    if (first_arg != cmd_args.end()) {
        if (std::string_view(cmd_args[1]).starts_with("-cc1")) {
            // FIXME: deal with EOL sentinels
            return execute_cc1_tool(cmd_args);
        }
    }

    // Handle options that need handling before the real command line parsing in
    // Driver::BuildCompilation()
    bool canonical_prefixes = has_canonical_prefixes_option(cmd_args);

    preprocess_vast_arguments(cmd_args);

    // FIXME: handle options that need handling before the real command line parsing
    std::string driver_path = get_executable_path(cmd_args[0], canonical_prefixes);

    // Not in the frontend mode - continue in the compiler driver mode.
    vast::cc::driver driver(driver_path, cmd_args, &execute_cc1_tool, canonical_prefixes);
    return driver.execute();
} catch (std::exception &e) {
    llvm::errs() << "error: " << e.what() << '\n';
    std::exit(1);
}
