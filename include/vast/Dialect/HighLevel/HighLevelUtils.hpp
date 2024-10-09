// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"

#include "vast/Util/Common.hpp"
#include <vast/Util/Symbols.hpp>
#include "vast/Util/TypeUtils.hpp"

#include <gap/coro/generator.hpp>

#include <ranges>

/* Contains common utilities often needed to work with hl dialect. */

namespace vast::hl {

    static inline gap::generator< mlir_type > get_field_types(auto op) {
        for (auto &&[_, type] : get_fields_info(op)) {
            co_yield type;
        }
    }

    gap::generator< core::field_info_t > get_fields_info(auto op) {
        for (auto &maybe_field : op.getOps()) {
            // Definition of nested structure, we ignore not a field.
            if (mlir::isa< core::aggregate_interface >(maybe_field)) {
                continue;
            }

            auto fld = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
            VAST_ASSERT(fld);
            co_yield { mlir::SymbolRefAttr::get(fld), fld.getType() };
        }
    }

    gap::generator< core::aggregate_interface > get_nested_declarations(auto op) {
        for (auto &maybe_field : op.getOps()) {
            if (auto casted = mlir::dyn_cast< core::aggregate_interface >(maybe_field)) {
                co_yield casted;
            }
        }
    }

    static inline gap::generator< mlir_type > field_types(hl::RecordType ty, operation op) {
        auto def = core::symbol_table::lookup< core::type_symbol >(op, ty.getName());
        VAST_CHECK(def, "Record type {} not present in the symbol table.", ty.getName());
        auto agg = mlir::dyn_cast_if_present< core::aggregate_interface >(def);
        VAST_CHECK(agg, "Record type symbol is not an aggregate.");
        return agg.getFieldTypes();
    }

    namespace detail {
        template< typename T >
        std::vector< T > to_vector(gap::generator< T > &&gen) {
            std::vector< T > result;
            std::ranges::copy(gen, std::back_inserter(result));
            return result;
        }
    } // namespace detail

    static inline auto field_index(string_ref name, core::aggregate_interface agg)
        -> std::optional< std::size_t >
    {
        for (const auto &[index, value] : llvm::enumerate(detail::to_vector(agg.getFieldsInfo()))) {
            if (value.name.getValue() == name) {
                return index;
            }
        }

        return std::nullopt;
    }

    walk_result users(hl::TypeDefOp op, auto scope, auto &&yield) {
        return type_users([&](mlir_type ty) {
            if (auto td = mlir::dyn_cast< TypedefType >(ty))
                return td.getName() == op.getSymName();
            return false;
        }, scope, std::forward< decltype(yield) >(yield));
    }

    walk_result users(hl::TypeDeclOp op, auto scope, auto &&yield) {
        return type_users([&](mlir_type ty) {
            if (auto rt = mlir::dyn_cast< RecordType >(ty))
                return rt.getName() == op.getSymName();
            return false;
        }, scope, std::forward< decltype(yield) >(yield));
    }

    walk_result users(core::aggregate_interface op, auto scope, auto &&yield) {
        return type_users([&](mlir_type ty) {
            if (auto rt = mlir::dyn_cast< RecordType >(ty))
                return rt.getName() == op.getDefinedName();
            return false;
        }, scope, std::forward< decltype(yield) >(yield));
    }

    walk_result users(hl::VarDeclOp var, auto scope, auto &&yield) {
        VAST_CHECK(var.hasGlobalStorage(), "Only global variables are supported");
        return scope.walk([&](DeclRefOp op) {
            return op.getName() == var.getSymbolName() ? yield(op) : walk_result::advance();
        });
    }

    walk_result users(core::FuncSymbolOpInterface fn, auto scope, auto &&yield) {
        return scope.walk([&](operation op) {
            if (auto call = mlir::dyn_cast< hl::CallOp >(op)) {
                return call.getCallee() == fn.getSymbolName() ? yield(op) : walk_result::advance();
            }

            if (auto ref = mlir::dyn_cast< hl::FuncRefOp >(op)) {
                return ref.getFunction() == fn.getSymbolName() ? yield(op) : walk_result::advance();
            }

            return walk_result::advance();
        });
    }

} // namespace vast::hl
