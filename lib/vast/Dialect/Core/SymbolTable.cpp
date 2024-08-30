// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Dialect/Core/SymbolTable.hpp"

#include <llvm/ADT/SetOperations.h>

#include <gap/core/ranges.hpp>

#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolTableInterface.hpp"

namespace vast::core {

    // checks if the symbol table can hold all symbol kinds of other_table
    bool subsumes(symbol_table_op_interface table, symbol_table_op_interface other_table) {
        auto self = table.symbol_kinds();
        return llvm::set_is_subset(
            other_table.symbol_kinds(), llvm::SmallDenseSet< symbol_kind >(self.begin(), self.end())
        );
    }

    auto immediate_unrecognized_nested_symbols(operation table) {
        auto not_recognized = [table](operation op) {
            auto self = mlir::cast< symbol_table_op_interface >(table);
            return !self.can_hold_operation(op);
        };

        return immediate_nested_symbols(table) | vws::filter(not_recognized);
    }

    auto nested_symbol_tables_view(operation root) {
        return gmw::operations(root) | gmw::isa< symbol_table_op_interface >;
    }

    auto addresses = vws::transform([](auto &v) { return std::addressof(v); });

    std::vector< operation > nested_symbol_tables(operation root) {
        return nested_symbol_tables_view(root) | addresses | gap::ranges::to< std::vector >;
    }


    auto nested_symbol_tables_with_unrecognized_symbols(operation root) {
        return nested_symbol_tables_view(root)
            | gmw::cast< symbol_table_op_interface >
            | vws::filter([rt = mlir::cast< symbol_table_op_interface >(root)] (auto st) {
                // Return true if st can contain symbols not kept in st,
                // but recognized by the root symbol table.
                return !subsumes(st, rt);
            });
    }

    gap::recursive_generator< operation > symbols_unrecognized_by_nested_symbol_tables(operation root) {
        VAST_ASSERT(mlir::isa< symbol_table_op_interface >(root));

        for (auto st : nested_symbol_tables_with_unrecognized_symbols(root)) {
            for (auto op : immediate_unrecognized_nested_symbols(st)) {
                co_yield op;
            }

            for ([[maybe_unused]] auto op : symbols_unrecognized_by_nested_symbol_tables(st)) {
                VAST_UNIMPLEMENTED_MSG("recursively yield the nested symbol tables of the nested symbol table");
            }
        }
    }

    void symbol_table::insert(symbol_kind kind, operation op) {
        VAST_ASSERT(can_hold_symbol_kind(kind));
        auto symbol_name = mlir::cast< symbol >(op).getSymbolName();
        VAST_ASSERT(symbol_tables.contains(kind));
        symbol_tables[kind][symbol_name].push_back(op);
    }

    string_ref symbol_attr_name() {
        return mlir::SymbolTable::getSymbolAttrName();
    }

    std::optional< symbol_table > get_effective_symbol_table_for(
        operation from, symbol_kind kind
    ) {
        while (from) {
            if (auto table = mlir::dyn_cast_if_present< SymbolTableOpInterface >(from)) {
                if (table.can_hold_symbol_kind(kind)) {
                    return table.materialize();
                }
            }
            from = from->getParentOp();
        }

        return std::nullopt;
    }

} // namespace vast::core
