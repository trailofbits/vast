// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/SymbolTable.h>
VAST_UNRELAX_WARNINGS


namespace vast::util
{
    // TODO(heno): rework to coroutines eventually
    void symbols(mlir::Operation *op, auto yield) {
        op->walk([&] (mlir::Operation *child) { 
            if (auto symbol = mlir::dyn_cast< mlir::SymbolOpInterface >(child)) {
                yield(symbol);
            }
        });
    }
} // namespace vast::util
