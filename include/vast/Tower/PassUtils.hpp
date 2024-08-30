// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Tower/Handle.hpp"

namespace vast::tw {

    // `mlir::Pass::printAsTextualPipeline` is not `const` so we cannot accept `const`
    // argument.
    std::string to_string(pass_ptr pass);

    std::string to_string(const conversion_passes_t &passes);

    // `mlir::PassManager` is really hard to move around, so we instead fill an existing
    // instance.
    void copy_passes(mlir::PassManager &pm, const conversion_passes_t &passes);

} // namespace vast::tw
