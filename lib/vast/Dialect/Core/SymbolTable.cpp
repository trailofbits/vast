// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Dialect/Core/SymbolTable.hpp"

#include "vast/Interfaces/SymbolInterface.hpp"
#include "vast/Interfaces/SymbolTableInterface.hpp"

namespace vast::core {

    symbol_table::symbol_table(operation symbol_table_op)
        : symbol_table_op(symbol_table_op)
    {
        VAST_ASSERT(symbol_table_op->hasTrait< ShadowingSymbolTable >());
        // for (auto &region : symbol_table_op->getRegions()) {
        //     for (auto &block : region) {
        //         for (auto &op : block) {
        //             if (auto symbol = mlir::dyn_cast< core::symbol_base >(op)) {
        //                 auto symbol_name = symbol.getSymbolName();
        //                 // auto symbol_id = symbol.getSymbolInterfaceID();
        //                 // split by symbol kinds here
        //                 // symbol_tables[symbol_id][symbol_name] = op;
        //             }
        //         }
        //     }
        // }
    }

    operation symbol_table::lookup(string_ref symbol) const {
        VAST_UNIMPLEMENTED;
    }


    operation symbol_table::lookup(string_attr symbol) const {
        return lookup(symbol.getValue());
    }

    string_ref symbol_attr_name() {
        return mlir::SymbolTable::getSymbolAttrName();
    }

    std::optional< symbol_table > get_effective_symbol_table_for(
        operation from, symbol_kind kind
    ) {
        while (from) {
            if (auto table = mlir::dyn_cast_if_present< VastSymbolTable >(from)) {
                if (table.holds_symbol_kind(kind)) {
                    return symbol_table(from);
                }
            }
            from = from->getParentOp();
        }

        return std::nullopt;
    }


} // namespace vast::core
