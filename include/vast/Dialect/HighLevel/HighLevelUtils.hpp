// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Interfaces/SymbolInterface.hpp"

#include "vast/Util/Common.hpp"

#include "gap/core/generator.hpp"

/* Contains common utilities often needed to work with hl dialect. */

namespace vast::hl
{
    static inline gap::generator< mlir_type > get_field_types(auto op) {
        for (auto [_, type] : get_fields_info(op))
            co_yield type;
    }

    static inline gap::generator< std::tuple< std::string, mlir_type > > get_fields_info(
        auto op
    ) {
        for (auto &maybe_field : op.getOps())
        {
            // Definition of nested structure, we ignore not a field.
            if (mlir::isa< AggregateTypeDefinitionInterface >(maybe_field))
                continue;

            auto field_decl = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
            VAST_ASSERT(field_decl);
            co_yield std::make_tuple(field_decl.getName().str(), field_decl.getType());
        }
    }

    static inline gap::generator< AggregateTypeDefinitionInterface > get_nested_declarations(
        auto op
    ) {
        for (auto &maybe_field : op.getOps())
            if (auto casted = mlir::dyn_cast< AggregateTypeDefinitionInterface >(maybe_field))
                co_yield casted;
    }

    // TODO(hl): This is a placeholder that works in our test cases so far.
    //           In general, we will need generic resolution for scoping that
    //           will be used instead of this function.
    static inline auto definition_of(mlir::Type t, vast_module module_op)
        -> AggregateTypeDefinitionInterface
    {
        auto type_name = hl::name_of_record(t);
        VAST_CHECK(type_name, "hl::name_of_record failed with {0}", t);

        AggregateTypeDefinitionInterface out;;
        auto walker = [&](AggregateTypeDefinitionInterface op) {
            if (op.getDefinedName() == type_name)
            {
                out = op;
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();

        };
        module_op->walk(walker);
        return out;
    }


    static inline auto field_types(mlir::Type t, vast_module module_op)
        -> gap::generator< mlir_type >
    {
        auto def = definition_of(t, module_op);
        VAST_CHECK(def, "Was not able to fetch definition of type: {0}", t);
        return def.getFieldTypes();
    }

    static inline hl::ImplicitCastOp implicit_cast_lvalue_to_rvalue(
        auto &rewriter, auto loc, auto lvalue_op
    ){
        auto lvalue_type = mlir::dyn_cast< hl::LValueType >(lvalue_op.getType());
        VAST_ASSERT(lvalue_type);
        return rewriter.template create< hl::ImplicitCastOp >(
            loc, lvalue_type.getElementType(),
            lvalue_op, hl::CastKind::LValueToRValue);
    }

    // Given record `root` emit `hl::RecordMemberOp` for each its member.
    static inline auto generate_ptrs_to_record_members(operation root, auto loc, auto &bld)
        -> gap::generator< hl::RecordMemberOp >
    {
        auto module_op = root->getParentOfType< vast_module >();
        VAST_ASSERT(module_op);
        auto def = definition_of(root->getResultTypes()[0], module_op);
        VAST_CHECK(def, "Was not able to fetch definition of type from: {0}", *root);

        for (const auto &[name, type] : def.getFieldsInfo())
        {
            VAST_ASSERT(root->getNumResults() == 1);
            auto as_val = root->getResult(0);
            // `hl.member` requires type to be an lvalue.
            auto wrap_type = hl::LValueType::get(module_op.getContext(), type);
            co_yield bld.template create< hl::RecordMemberOp >(loc, wrap_type, as_val, name);
        }
    }

    // Given record `root` emit `hl::RecordMemberOp` casted as rvalue for each
    // its member.
    static inline auto generate_values_of_record_members(operation root, auto &bld)
        -> gap::generator< hl::ImplicitCastOp >
    {
        for (auto member_ptr : generate_ptrs_to_members(root, bld)) {
            co_yield implicit_cast_lvalue_to_rvalue(bld, member_ptr->getLoc(), member_ptr);
        }
    }

    static inline std::optional< std::size_t > field_idx(
        llvm::StringRef name, AggregateTypeDefinitionInterface decl
    ) {
        std::size_t idx = 0;
        // `llvm::enumerate` is unhappy when coroutine is passed in.
        for (const auto &[field_name, _] : decl.getFieldsInfo())
        {
            if (field_name == name)
                return { idx };
            ++idx;
        }
        return {};
    }
}
