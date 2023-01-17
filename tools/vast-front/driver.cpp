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

#include "vast/Frontend/Common.hpp"
#include "vast/Frontend/Driver.hpp"

// main frontend method. Lives inside cc1_main.cpp
namespace vast::cc {
    extern int cc1(const vast_args & vargs, argv_t argv, arg_t tool, void *main_addr);
} // namespace vast::cc

VAST_RELAX_WARNINGS
std::string get_executable_path(vast::cc::arg_t tool) {
    // This just needs to be some symbol in the binary
    void *p = (void *) (intptr_t) get_executable_path;
    return llvm::sys::fs::getMainExecutable(tool, p);
}
VAST_UNRELAX_WARNINGS


static int execute_cc1_tool(vast::cc::vast_args &vargs, vast::cc::argv_storage &ccargs) {
    // If we call the cc1 tool from the clangDriver library (through
    // Driver::CC1Main), we need to clean up the options usage count. The options
    // are currently global, and they might have been used previously by the
    // driver.
    llvm::cl::ResetAllOptionOccurrences();

    llvm::BumpPtrAllocator pointer_allocator;
    llvm::StringSaver saver(pointer_allocator);
    llvm::cl::ExpandResponseFiles(
        saver, &llvm::cl::TokenizeGNUCommandLine, ccargs, /* MarkEOLs */ false
    );

    llvm::StringRef tool = ccargs[1];

    VAST_RELAX_WARNINGS
    void *get_executable_path_ptr = (void *) (intptr_t) get_executable_path;
    VAST_UNRELAX_WARNINGS

    if (tool == "-cc1") {
        auto ccargs_ref = llvm::makeArrayRef(ccargs).slice(2);
        return vast::cc::cc1(vargs, ccargs_ref, ccargs[0], get_executable_path_ptr);
    }

    llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
                 << "Valid tools include '-cc1'.\n";
    return 1;
}

int main(int argc, char **argv) try {
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);
    vast::cc::argv_storage cmd_args(argv, argv + argc);

    if (llvm::sys::Process::FixupStandardFileDescriptors()) {
        return 1;
    }

    llvm::InitializeAllTargets();

    llvm::BumpPtrAllocator pointer_allocator;
    llvm::StringSaver saver(pointer_allocator);

    auto [vargs, cc_args] = vast::cc::filter_args(cmd_args);

    // FIXME: deal with CL mode

    // Check if vast-front is in the frontend mode
    auto first_arg = llvm::find_if(llvm::drop_begin(cc_args), [] (auto a) { return a != nullptr; });
    if (first_arg != cc_args.end()) {
        if (std::string_view(cc_args[1]).starts_with("-cc1")) {
            // FIXME: deal with EOL sentinels
            return execute_cc1_tool(vargs, cc_args);
        }
    }

    throw vast::cc::compiler_error( "unsupported non -cc1 mode" );
    // FIXME: handle options that need handling before the real command line parsing

    // std::string driver_path = get_executable_path(args[0]);

    // // Not in the frontend mode - continue in the compiler driver mode.
    // vast::cc::driver driver(driver_path, args);
    // auto comp = driver.make_compilation(args);
    // vast::cc::failing_commands failing;

    // // Run the driver
    // driver.execute(*comp, failing);

    // int res = 1;
    // for (const auto &[cmd_res, cmd] : failing) {
    //     res = (!res) ? cmd_res : res;

    //     // If result status is < 0 (e.g. when sys::ExecuteAndWait returns -1),
    //     // then the driver command signalled an error. On Windows, abort will
    //     // return an exit code of 3. In these cases, generate additional diagnostic
    //     // information if possible.
    //     if (win_32 ? cmd_res == 3 : cmd_res < 0) {
    //         // driver.generate_comp_diagnostics(*c, *failing);
    //         break;
    //     }
    // }

    // diags.client->finish();

    // // If we have multiple failing commands, we return the result of the first
    // // failing command.
    // return res;
} catch (vast::cc::compiler_error &e) {
    llvm::errs() << "vast-cc error: " << e.what() << '\n';
    std::exit(e.exit);
} catch (std::exception &e) {
    llvm::errs() << "error: " << e.what() << '\n';
    std::exit(1);
}
