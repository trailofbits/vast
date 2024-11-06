// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

namespace vast {

    std::unique_ptr< mlir::Pass > createHLToParserPass();

} // namespace vast
