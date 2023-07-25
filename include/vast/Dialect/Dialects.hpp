// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"
#include "vast/Dialect/Meta/MetaDialect.hpp"
#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"

#include "vast/Util/Common.hpp"

namespace vast {

    inline void registerAllDialects(mlir::DialectRegistry &registry) {
        registry.insert<
            vast::hl::HighLevelDialect,
            vast::ll::LowLevelDialect,
            vast::meta::MetaDialect,
            vast::core::CoreDialect,
            vast::us::UnsupportedDialect
        >();
    }

    inline void registerAllDialects(mcontext_t &mctx) {
        mlir::DialectRegistry registry;
        vast::registerAllDialects(registry);
        mctx.appendDialectRegistry(registry);
    }

} // namespace vast
