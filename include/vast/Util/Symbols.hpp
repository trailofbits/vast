// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/SymbolTable.h>
VAST_UNRELAX_WARNINGS


namespace vast::util
{
    // TODO(heno): rework to coroutines eventually
    void symbols(mlir::Operation *op, auto yield);

    void symbols(mlir::Region &reg, auto yield) {
        for (auto &op :reg.getOps()) {
            if (auto symbol = mlir::dyn_cast< mlir::SymbolOpInterface >( &op ) ) {
                yield(symbol);
            }

            op.walk([&] (mlir::Operation *child) { symbols(child, yield); });
        }
    }

    void symbols(mlir::Operation *op, auto yield) {
        for (auto &reg : op->getRegions()) {
            symbols(reg, yield);
        }
    }
} // namespace vast::util
