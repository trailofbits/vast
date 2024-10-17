// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "vast/Util/Common.hpp"

/// Include the generated interface declarations.
#include "vast/Dialect/Core/Interfaces/DesugarTypeInterface.h.inc"

namespace vast::core {
    mlir_type desugar_type(mlir_type type);
} // namespace vast::core
