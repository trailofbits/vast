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
#include <gap/coro/recursive_generator.hpp>
#include <gap/mlir/views.hpp>

namespace vast::core {

    namespace gmw = gap::mlir::views;
    namespace rns = std::ranges;
    namespace vws = std::ranges::views;

    using string_attr = mlir::StringAttr;

    struct symbol_table;

    template< symbol_op_interface symbol_kind >
    [[nodiscard]] std::optional< symbol_table > get_effective_symbol_table_for(operation from);

    template< symbol_op_interface symbol_kind >
    [[nodiscard]] std::optional< symbol_table > get_next_effective_symbol_table_for(operation from);

    template< util::flat_list symbols_list >
    bool can_hold_symbol_kind(symbol_kind kind) {
        if constexpr ( symbols_list::empty ) {
            return false;
        } else if (is_symbol_kind< typename symbols_list::head >(kind)) {
            return true;
        } else {
            return can_hold_symbol_kind< typename symbols_list::tail >(kind);
        }
    }

    template< util::list_of_lists symbols_lists >
    bool can_hold_symbol_kind(symbol_kind kind) {
        return can_hold_symbol_kind< util::flatten< symbols_lists > >(kind);
    }

    template< util::flat_list symbols_list >
    bool can_hold_operation(operation op) {
        if constexpr ( symbols_list::empty ) {
            return false;
        } else if (mlir::isa< typename symbols_list::head >(op)) {
            return true;
        } else {
            return can_hold_operation< typename symbols_list::tail >(op);
        }
    }

    template< util::list_of_lists symbols_lists >
    bool can_hold_operation(operation op) {
        return can_hold_operation< util::flatten< symbols_lists > >(op);
    }

    namespace detail {
        template< util::flat_list symbols_list >
        void symbol_kinds(auto &result) {
            if constexpr ( symbols_list::empty ) {
                return;
            } else {
                result.push_back(get_symbol_kind< typename symbols_list::head >);
                symbol_kinds< typename symbols_list::tail >(result);
            }
        }

        template< util::list_of_lists symbols_lists >
        void symbol_kinds(auto &result) {
            symbol_kinds< util::flatten< symbols_lists > >(result);
        }
    } // namespace detail

    template< util::list_of_lists symbols_lists >
    llvm::SmallVector< symbol_kind > symbol_kinds() {
        constexpr auto size = util::flatten< symbols_lists >::size;
        llvm::SmallVector< symbol_kind, size > result;
        detail::symbol_kinds< symbols_lists >(result);
        return result;
    }

    static inline auto immediate_nested_symbols(operation table) {
        return gmw::operations(table) | gmw::filter_cast< symbol_base >;
    }

    gap::recursive_generator< operation > symbols_unrecognized_by_nested_symbol_tables(operation root);


    struct symbol_table
    {
        using single_symbol_kind_table = llvm::DenseMap< string_ref, operation >;

        template< util::list_of_lists symbols_lists >
        explicit symbol_table(std::in_place_type_t< symbols_lists >, operation symbol_table_op)
            : symbol_table_op(symbol_table_op)
        {
            using recognized_symbols_list = util::flatten< symbols_lists >;
            symbol_tables.reserve( recognized_symbols_list::size );
            setup_symbol_tables< recognized_symbols_list >();

            insert_nested_symbols< recognized_symbols_list >(symbol_table_op);
        }

        [[nodiscard]] operation lookup(string_ref symbol) const;
        [[nodiscard]] operation lookup(string_attr symbol) const;

        [[nodiscard]] operation lookup(operation from, string_ref symbol);

        template< symbol_op_interface symbol_kind >
        [[nodiscard]] static operation lookup(operation from, string_attr symbol);

        template< symbol_op_interface symbol_kind >
        [[nodiscard]] static operation lookup(operation from, string_ref symbol);

        template< symbol_op_interface symbol_kind >
        [[nodiscard]] operation lookup(string_attr symbol) const;

        template< util::flat_list symbols_list >
        void try_insert(operation op) {
            if constexpr ( symbols_list::empty ) {
                return;
            } else if (mlir::isa< typename symbols_list::head >(op)) {
                return insert(get_symbol_kind< typename symbols_list::head >, op);
            } else {
                return try_insert< typename symbols_list::tail >(op);
            }
        }

        template< util::list_of_lists symbol_lists >
        void try_insert(operation op) {
            try_insert< util::flatten< symbol_lists > >(op);
        }

        bool can_hold_symbol_kind(symbol_kind kind) const {
            return symbol_tables.contains(kind);
        }

        operation symbol_table_op;

      protected:

        template< util::flat_list symbols_list >
        void setup_symbol_tables() {
            if constexpr ( symbols_list::empty ) {
                return;
            } else {
                // create an empty table for recognized symbol kind
                symbol_tables.try_emplace(
                    get_symbol_kind< typename symbols_list::head >,
                    single_symbol_kind_table{}
                );

                // continue with the rest of the recognized symbols
                setup_symbol_tables< typename symbols_list::tail >();
            }
        }

        template< util::flat_list symbols_list >
        void insert_nested_symbols(operation op) {
            // insert all recognized imediate symbols into the symbol tables
            for (auto symbol : immediate_nested_symbols(op)) {
                try_insert< symbols_list >(symbol);
            }

            // insert all recognized symbols from nested scopes
            // not recognized by the symbol table of nested scope
            for (auto symbol : symbols_unrecognized_by_nested_symbol_tables(op)) {
                try_insert< symbols_list >(symbol);
            }
        }

        void insert(symbol_kind kind, operation op);

        llvm::DenseMap< symbol_kind, single_symbol_kind_table > symbol_tables;
    };


    template< symbol_op_interface symbol_kind >
    operation symbol_table::lookup(operation from, string_ref symbol) {
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
    operation symbol_table::lookup(operation from, string_attr symbol) {
        return lookup< symbol_kind >(from, symbol.getValue());
    }

    template< symbol_op_interface symbol_kind >
    operation symbol_table::lookup(string_attr symbol) const {
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
