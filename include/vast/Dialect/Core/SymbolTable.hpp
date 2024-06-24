// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/OpDefinition.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreTraits.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Interfaces/SymbolInterface.hpp"

#include <gap/coro/generator.hpp>

namespace vast::core {

    using string_attr     = mlir::StringAttr;

    struct symbol_table;

    template< symbol_op_interface symbol_kind >
    [[nodiscard]] std::optional< symbol_table > get_effective_symbol_table_for(operation from);

    template< symbol_op_interface symbol_kind >
    [[nodiscard]] std::optional< symbol_table > get_next_effective_symbol_table_for(operation from);

    struct symbol_table
    {
        explicit symbol_table(operation symbol_table_op);

        [[nodiscard]] operation lookup(string_ref symbol) const;
        [[nodiscard]] operation lookup(string_attr symbol) const;

        operation symbol_table_op;
    };

    template< typename recognized_symbols_list >
    struct constrained_symbol_table : symbol_table {};


    // Rename to table that categoriezes symbols by kind
    struct grouped_symbol_table
    {
        explicit grouped_symbol_table(operation symbol_table_op);

        template< symbol_op_interface symbol_kind >
        [[nodiscard]] static operation lookup(operation from, string_attr symbol);

        template< symbol_op_interface symbol_kind >
        [[nodiscard]] static operation lookup(operation from, string_ref symbol);

        template< symbol_op_interface symbol_kind >
        [[nodiscard]] operation lookup(string_attr symbol) const;

        template< symbol_op_interface symbol_kind >
        [[nodiscard]] operation table() const;

        operation symbol_table_op;

      private:

        using constrained_symbol_table = llvm::DenseMap< string_ref, operation >;
    };

    template< symbol_op_interface symbol_kind >
    operation grouped_symbol_table::lookup(operation from, string_ref symbol) {
        auto table = get_effective_symbol_table_for< symbol_kind >(from);
        VAST_CHECK(table, "No effective symbol table found.");

        while (table) {
            if (auto result = table->lookup(symbol))
                return result;
            table = get_next_effective_symbol_table_for< symbol_kind >(table->symbol_table_op);
        }

        return {};
    }

    template< symbol_op_interface symbol_kind >
    operation grouped_symbol_table::lookup(operation from, string_attr symbol) {
        return lookup< symbol_kind >(from, symbol.getValue());
    }

    template< symbol_op_interface symbol_kind >
    operation grouped_symbol_table::lookup(string_attr symbol) const {
        return lookup< symbol_kind >(symbol_table_op, symbol);
    }

    std::optional< symbol_table > get_effective_symbol_table_for(operation from, symbol_kind kind);

    template< symbol_op_interface symbol_kind >
    std::optional< symbol_table > get_effective_symbol_table_for(operation from) {
        return get_effective_symbol_table_for(from, get_symbol_kind< symbol_kind >);
    }

    template< symbol_op_interface symbol_kind >
    std::optional< symbol_table > get_next_effective_symbol_table_for(operation from) {
        return get_effective_symbol_table_for< symbol_kind >(from->getParentOp());
    }

    //
    // Name of the symbol attribute to be used in operations declaring symbols.
    //
    // At the moment, this wraps `mlir::SymbolTable::getSymbolAttrName()` but it
    // allows us easily replace it in the future when we need to distinguish
    // various symbol attributes.
    //
    string_ref symbol_attr_name();

} // namespace vast::core
