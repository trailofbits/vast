// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Driver/Compilation.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Driver/Driver.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Diagnostics.hpp"

namespace vast::cc {

    //
    // driver wrappers
    //
    using clang_driver = clang::driver::Driver;

    struct driver {
        driver(const std::string &path, llvm::SmallVectorImpl<const char *> &argv)
            : diag(argv), drv(path, llvm::sys::getDefaultTargetTriple(), diag.engine, "vast compiler")
        {
            // FIXME: use SetInstallDir(Args, TheDriver, CanonicalPrefixes);
            // FIXME: set target and mode
        }

        diagnostics diag;
        clang_driver drv;
    };

} // namespace vast::cc
