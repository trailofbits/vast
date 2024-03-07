// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Interfaces/SymbolInterface.hpp"

#include "vast/Util/TypeUtils.hpp"

#include "vast/Util/Common.hpp"

#include <gap/core/generator.hpp>

#include <ranges>

/* Contains common utilities often needed to work with hl dialect. */

namespace vast::hl {

    using aggregate_interface = AggregateTypeDefinitionInterface;

    static inline gap::generator< mlir_type > get_field_types(auto op) {
        for (auto &&[_, type] : get_fields_info(op)) {
            co_yield type;
        }
    }

    gap::generator< field_info_t > get_fields_info(auto op) {
        for (auto &maybe_field : op.getOps()) {
            // Definition of nested structure, we ignore not a field.
            if (mlir::isa< aggregate_interface >(maybe_field)) {
                continue;
            }

            auto field_decl = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
            VAST_ASSERT(field_decl);
            co_yield { field_decl.getName().str(), field_decl.getType() };
        }
    }

    gap::generator< aggregate_interface > get_nested_declarations(auto op) {
        for (auto &maybe_field : op.getOps()) {
            if (auto casted = mlir::dyn_cast< aggregate_interface >(maybe_field)) {
                co_yield casted;
            }
        }
    }

    // TODO(hl): This is a placeholder that works in our test cases so far.
    //           In general, we will need generic resolution for scoping that
    //           will be used instead of this function.
    aggregate_interface definition_of(mlir_type ty, auto scope) {
        auto type_name = hl::name_of_record(ty);
        VAST_CHECK(type_name, "hl::name_of_record failed with {0}", ty);

        aggregate_interface out;
        auto walker = [&](aggregate_interface op) {
            if (op.getDefinedName() == type_name) {
                out = op;
                return walk_result::interrupt();
            }
            return mlir::WalkResult::advance();
        };
        scope->walk(walker);
        return out;
    }

    gap::generator< mlir_type >  field_types(mlir_type ty, auto scope) {
        auto def = definition_of(ty, scope);
        VAST_CHECK(def, "Was not able to fetch definition of type: {0}", ty);
        return def.getFieldTypes();
    }

    hl::ImplicitCastOp implicit_cast_lvalue_to_rvalue(auto &rewriter, auto loc, auto lvalue_op) {
        auto value_type = mlir::dyn_cast< hl::LValueType >(lvalue_op.getType());
        VAST_ASSERT(value_type);
        return rewriter.template create< hl::ImplicitCastOp >(
            loc, value_type.getElementType(), lvalue_op, hl::CastKind::LValueToRValue
        );
    }

    // Given record `root` emit `hl::RecordMemberOp` for each its member.
    auto generate_ptrs_to_record_members(operation root, auto loc, auto &bld)
        ->  gap::generator< hl::RecordMemberOp >
    {
        auto scope = root->getParentOfType< vast_module >();
        VAST_ASSERT(scope);
        VAST_ASSERT(root->getNumResults() == 1);
        auto def = definition_of(root->getResultTypes()[0], scope);
        VAST_CHECK(def, "Was not able to fetch definition of type from: {0}", *root);

        for (const auto &[name, type] : def.getFieldsInfo()) {
            auto as_val = root->getResult(0);
            // `hl.member` requires type to be an lvalue.
            auto wrap_type = hl::LValueType::get(scope.getContext(), type);
            co_yield bld.template create< hl::RecordMemberOp >(loc, wrap_type, as_val, name);
        }
    }

    // Given record `root` emit `hl::RecordMemberOp` casted as rvalue for each
    // its member.
    auto generate_values_of_record_members(operation root, auto &bld)
        -> gap::generator< hl::ImplicitCastOp >
    {
        for (auto member_ptr : generate_ptrs_to_members(root, bld)) {
            co_yield implicit_cast_lvalue_to_rvalue(bld, member_ptr->getLoc(), member_ptr);
        }
    }

    namespace detail
    {
        template <typename T>
        std::vector<T> to_vector(gap::generator<T> &&gen) {
            std::vector<T> result;
            std::ranges::copy(gen, std::back_inserter(result));
            return result;
        }
    } // namespace detail

    static inline auto field_index(string_ref name, aggregate_interface agg)
        -> std::optional< std::size_t >
    {
        for (const auto &field : llvm::enumerate(detail::to_vector(agg.getFieldsInfo()))) {
            if (field.value().name == name) {
                return field.index();
            }
        }

        return std::nullopt;
    }

    template< typename yield_t >
    walk_result users(aggregate_interface agg, auto scope, yield_t &&yield) {
        return type_users([&] (mlir_type ty) {
            if (auto rt = mlir::dyn_cast< RecordType >(ty))
                return rt.getName() == agg.getDefinedName();
            return false;
        }, scope, std::forward< yield_t >(yield));
    }

    template< typename yield_t >
    walk_result users(hl::TypeDefOp def, auto scope, yield_t &&yield) {
        return type_users([&] (mlir_type ty) {
            if (auto td = mlir::dyn_cast< TypedefType >(ty))
                return td.getName() == def.getName();
            return false;
        }, scope, std::forward< yield_t >(yield));
    }


} // namespace vast::hl
