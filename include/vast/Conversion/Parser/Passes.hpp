// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/server/server.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

namespace vast {

    std::unique_ptr< mlir::Pass > createHLToParserPass();
    std::unique_ptr< mlir::Pass > createParserReconcileCastsPass();
    std::unique_ptr< mlir::Pass > createParserSourceToSarifPass();

    // Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Conversion/Parser/Passes.h.inc"

} // namespace vast
