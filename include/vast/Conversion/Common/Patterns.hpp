// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

namespace vast {

    using pattern_rewriter = mlir::PatternRewriter;

    template< typename op_t >
    using operation_rewrite_pattern = mlir::OpRewritePattern< op_t >;

    template< typename op_t >
    using operation_conversion_pattern = mlir::OpConversionPattern< op_t >;

} // namespace vast
