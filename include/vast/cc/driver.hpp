// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
VAST_UNRELAX_WARNINGS

namespace vast::cc {

    // struct diagnostics_wrapper {

    // };

    // struct clang_driver : diagnostics_wrapper, clang::driver::Driver {
    //     clang_driver( const char* tool = "divcc" )
    //         : clang::driver::Driver( tool, LLVM_HOST_TRIPLE, diag.engine )
    //     {}
    // };

} // namespace vast::cc
