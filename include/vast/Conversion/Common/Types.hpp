// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

namespace vast {

    using pattern_rewriter = mlir::PatternRewriter;

    using conversion_rewriter = mlir::ConversionPatternRewriter;

    using conversion_target = mlir::ConversionTarget;
} // namespace vast
