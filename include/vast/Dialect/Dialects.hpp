// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/Meta/MetaDialect.hpp"

#include "vast/Util/Common.hpp"

namespace vast {

    inline void registerAllDialects(mlir::DialectRegistry &registry) {
        registry.insert<
            vast::hl::HighLevelDialect,
            vast::meta::MetaDialect
        >();
    }

    inline void registerAllDialects(MContext &mctx) {
        mlir::DialectRegistry registry;
        vast::registerAllDialects(registry);
        mctx.appendDialectRegistry(registry);
    }

} // namespace vast
