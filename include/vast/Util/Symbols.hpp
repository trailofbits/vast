// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/SymbolTable.h>
VAST_UNRELAX_WARNINGS


namespace vast::util
{
    // TODO(heno): rework to coroutines eventually
    void symbols(mlir::Operation *mod, auto yield) {

        auto walk_symbols = [&](mlir::Operation *table, bool) {
            for (auto &op : table->getRegion(0).getOps()) {
                if (auto symbol = mlir::dyn_cast< mlir::SymbolOpInterface >( &op ) )
                    yield(symbol);
                op.walk([&] (mlir::Operation *child) { symbols(child, yield); });
            }
        };

        mlir::SymbolTable::walkSymbolTables(mod, false, walk_symbols);
    }
} // namespace vast::util
