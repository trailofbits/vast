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

    //
    // direct symbol uses
    //

    auto direct_regions(region_ptr root) {
        return vws::single(root);
    }

    auto direct_regions(operation root) -> gap::generator< region_ptr > {
        for (auto &region : root->getRegions()) {
            co_yield &region;
        }
    }

    using symbol_use = mlir::SymbolTable::SymbolUse;

    string_attr get_symbol_name(operation op) {
        return op->getAttrOfType< string_attr >(symbol_attr_name());
    }

    // FIXME: This is not nice. Can this be inferred from the symbol reference directly?
    enum class reference_kind { var, type, func, label, member, enum_constant, elaborated_type };

    reference_kind get_reference_kind(symbol_ref_attr attr) {
        if (mlir::isa< var_symbol_ref_attr >(attr))
            return reference_kind::var;
        if (mlir::isa< type_symbol_ref_attr >(attr))
            return reference_kind::type;
        if (mlir::isa< func_symbol_ref_attr >(attr))
            return reference_kind::func;
        if (mlir::isa< label_symbol_ref_attr >(attr))
            return reference_kind::label;
        if (mlir::isa< member_var_symbol_ref_attr >(attr))
            return reference_kind::member;
        if (mlir::isa< enum_constant_symbol_ref_attr >(attr))
            return reference_kind::enum_constant;
        if (mlir::isa< elaborated_type_symbol_ref_attr >(attr))
            return reference_kind::elaborated_type;

        VAST_UNREACHABLE("unrecognized reference kind");
    }

    reference_kind get_reference_kind(operation symbol) {
        if (mlir::isa< var_symbol >(symbol))
            return reference_kind::var;
        if (mlir::isa< type_symbol >(symbol))
            return reference_kind::type;
        if (mlir::isa< func_symbol >(symbol))
            return reference_kind::func;
        if (mlir::isa< label_symbol >(symbol))
            return reference_kind::label;
        if (mlir::isa< member_symbol >(symbol))
            return reference_kind::member;
        if (mlir::isa< enum_constant_symbol >(symbol))
            return reference_kind::enum_constant;
        if (mlir::isa< elaborated_type_symbol >(symbol))
            return reference_kind::elaborated_type;

        VAST_UNREACHABLE("unrecognized reference kind");
    }

    bool is_reference_of(symbol_ref_attr attr, operation symbol) {
        return get_reference_kind(attr) == get_reference_kind(symbol);
    }

    symbol_ref_attr get_symbol_ref_attr(operation op, operation symbol) {
        symbol_ref_attr result;
        op->getAttrDictionary().walk< mlir::WalkOrder::PreOrder >(
            [&] (symbol_ref_attr attr) {
                if (!is_reference_of(attr, symbol)) {
                    return mlir::WalkResult::skip();
                }

                if (attr.getRootReference() == get_symbol_name(symbol)) {
                    result = attr;
                    return mlir::WalkResult::interrupt();
                }

                // Don't walk nested references.
                return mlir::WalkResult::skip();
            }
        );

        return result;
    }

    struct symbol_scope {
        // The first effective operation in the scope
        // allows to reduce the search space in the region.
        //
        // Can be used if definition of symbol is in middle of region, we want
        // to look at references only after the definition.
        //
        // If scope_begin is not set, the scope is the region itself.
        operation scope_begin;
        region_ptr scope;
    };

    gap::generator< operation > operations(symbol_scope region) {
        // TBD: use proper dominance analysis
        for (auto &bb : *region.scope) {
            for (auto &op : bb) {
                if (&bb != region.scope_begin->getBlock()) {
                    co_yield &op;
                } else if (!region.scope_begin || region.scope_begin->isBeforeInBlock(&op)) {
                    co_yield &op;
                }
            }
        }
    }

    std::optional< symbol_scope > constrain_ancestor_scope(operation scope, operation symbol) {
        if (symbol->getParentRegion()->isAncestor(scope->getParentRegion()))
            return symbol_scope{ symbol, symbol->getParentRegion() };
        return std::nullopt;
    }

    std::optional< symbol_scope > constrain_ancestor_scope(region_ptr scope, operation symbol) {
        if (symbol->getParentRegion()->isAncestor(scope))
            return symbol_scope{ symbol, scope };
        return std::nullopt;
    }

    gap::generator< symbol_scope > symbol_scopes(auto scope, operation symbol) {
        // If symbol is defined in ancestor region of scope, return the most immediate
        // region and constrain it to scope from symbol definition ownwards
        if (auto constrained = constrain_ancestor_scope(scope, symbol)) {
            co_yield *constrained;
        } else {
            // Else symbol is defined above scope, therefore references can be anywhere in scope
            for (auto region : direct_regions(scope)) {
                co_yield symbol_scope{ nullptr, region };
            }
        }
    }

    auto direct_symbol_uses_in_scope(operation symbol, symbol_scope scope)
        -> gap::generator< symbol_use >
    {
        for (auto op : operations(scope)) {
            if (auto symbol_ref = get_symbol_ref_attr(op, symbol)) {
                co_yield symbol_use{ op, symbol_ref };
            }
        }
    }

    auto direct_symbol_uses_in_scope(operation symbol, auto scope)
        -> gap::generator< symbol_use >
    {
        for (auto scope : symbol_scopes(scope, symbol)) {
            for (auto use : direct_symbol_uses_in_scope(symbol, scope)) {
                co_yield use;
            }
        }
    }


    namespace detail {

        symbol_use_range get_direct_symbol_uses_impl(operation symbol, auto root) {
            VAST_ASSERT(symbol);
            std::vector< symbol_use > uses;

            for (auto use : direct_symbol_uses_in_scope(symbol, root)) {
                uses.push_back(use);
            }

            return symbol_use_range(std::move(uses));
        }

    } // namespace detail

    symbol_use_range symbol_table::get_direct_symbol_uses(operation symbol, operation scope) {
        return detail::get_direct_symbol_uses_impl(symbol, scope);
    }

    symbol_use_range symbol_table::get_direct_symbol_uses(operation symbol, region_ptr scope) {
        return detail::get_direct_symbol_uses_impl(symbol, scope);
    }

    //
    // symbol uses
    //

    namespace detail {

        symbol_use_range get_symbol_uses_impl(operation symbol, auto scope) {
            VAST_ASSERT(symbol);
            std::vector< symbol_use > uses;

            auto symbol_region = symbol->getParentRegion();
            scope->walk([&](operation op) {
                if (op->getNumRegions() == 0)
                    return mlir::WalkResult::skip();

                bool skip = true;
                for (auto &region : op->getRegions()) {
                    if (symbol_region->isAncestor(&region)) {
                        for (auto use : direct_symbol_uses_in_scope(symbol, &region)) {
                            uses.push_back(use);
                        }
                        skip = false;
                    }
                }

                return skip ? mlir::WalkResult::skip() : mlir::WalkResult::advance();
            });

            return symbol_use_range(std::move(uses));
        }

    } // namespace detail

    symbol_use_range symbol_table::get_symbol_uses(operation symbol, operation scope) {
        return detail::get_symbol_uses_impl(symbol, scope);
    }

    symbol_use_range symbol_table::get_symbol_uses(operation symbol, region_ptr scope) {
        return detail::get_symbol_uses_impl(symbol, scope);
    }


    symbol_use_range get_symbol_uses(operation symbol, operation scope) {
        return symbol_table::get_symbol_uses(symbol, scope);
    }

} // namespace vast::core
