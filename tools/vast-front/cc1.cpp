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
#include <clang/Driver/DriverDiagnostic.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/TargetSelect.h>
VAST_UNRELAX_WARNINGS

int cc1_main(llvm::ArrayRef<const char *> argv, const char *argv0) {
    throw std::runtime_error("Unscupported cc1 frontend mode");
}
