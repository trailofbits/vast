//
// Copyright (c) 2022, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.
//

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/VirtualFileSystem.h>
VAST_UNRELAX_WARNINGS

#include "vast/cc/driver.hpp"

/*
 * This is the user-facing tool that forms the frontend for vast-cc. It's a
 * C/C++ compiler based on Clang whose main goal is to aid VAST MLIR-based
 * verification. To this end, it is capable of generating a whole-program binary
 * containing an additional .mlirbc section that is used to store MLIR bitcode
 * representation of the binary. It also handles linking of compilation units,
 * appending the bitcode sections where necessary.
 *
 * One of the aims of the project is to be a drop-in replacement for a more
 * traditional C/C++ toolchain such as GCC or Clang/LLVM. Therefore, the
 * user-facing CLI is basically the same.
 */

// main frontend method. Lives inside fc1_main.cpp
extern int cc1_main(llvm::ArrayRef<const char *> argv, const char *argv0);

std::string get_executable_path(const char *argv0) {
    // This just needs to be some symbol in the binary
    void *p = (void *) (intptr_t) get_executable_path;
    return llvm::sys::fs::getMainExecutable(argv0, p);
}

static int execute_cc1_tool(llvm::ArrayRef<const char *> args) {
    llvm::StringRef tool = args[1];
    if (tool == "-cc1") {
        return cc1_main(args.slice(2), args[0]);
    }

    llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
                 << "Valid tools include '-cc1'.\n";
    return 1;
}

int main(int argc, char **argv) try {
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);
    llvm::SmallVector< const char *, 256 > args(argv, argv + argc);

    std::string driver_path = get_executable_path(args[0]);

    // Check if vast-front is in the frontend mode
    if (std::next(args.begin()) != args.end()) {
        if (std::string_view(args[1]).starts_with("-cc1")) {
            return execute_cc1_tool(llvm::makeArrayRef(args));
        }
    }

    // Not in the frontend mode - continue in the compiler driver mode.

    // Create DiagnosticsEngine for the compiler driver

    // Prepare the driver

    // Run the driver
    int res = 1;

    // If we have multiple failing commands, we return the result of the first
    // failing command.
    return res;
} catch (std::exception &e) {
    llvm::errs() << "error: " << e.what() << '\n';
    std::exit(1);
}
