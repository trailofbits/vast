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
    template< typename T >
    gap::generator< T > top_level_ops(vast_module module_op)
    {
        auto body = module_op.getBody();
        if (!body)
            co_return;

        for (auto &op : *body)
            if (auto casted = mlir::dyn_cast< T >(op))
                co_yield casted;
    }

    using type_generator = gap::generator< mlir::Type >;
    using value_generator = gap::generator< mlir::Value >;

    template< typename op_t >
    concept defines_field_types = std::is_same_v< hl::StructDeclOp, op_t > ||
                                  std::is_same_v< hl::UnionDeclOp, op_t >;

    template< defines_field_types op_t >
    static inline gap::generator< hl::FieldDeclOp > field_defs(op_t op)
    {
        for (auto &maybe_field : op.getOps())
        {
            // TODO(hl): So normally only `hl.field` should be present here,
            //           but currently also re-declarations of nested structures
            //           are here - add hard fail if the conversion fails in the future.
            if (mlir::isa< hl::StructDeclOp >(maybe_field))
                continue;

            auto field_decl = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
            VAST_ASSERT(field_decl);
            co_yield field_decl;
        }
    }

    template< defines_field_types op_t >
    static inline type_generator field_types(op_t op)
    {
        for (auto def : field_defs(op))
            co_yield def.getType();
    }

    // TODO(hl): This is a placeholder that works in our test cases so far.
    //           In general, we will need generic resolution for scoping that
    //           will be used instead of this function.
    template< defines_field_types op_t = hl::StructDeclOp >
    static inline auto definition_of(mlir::Type t, vast_module module_op)
        -> std::optional< op_t >
    {
        auto type_name = hl::name_of_record(t);
        VAST_CHECK(type_name, "hl::name_of_record failed with {0}", t);
        for (auto op : top_level_ops< op_t >(module_op))
            if (op.getName() == *type_name)
                return { op };
        return {};
    }

    static inline auto type_decls(hl::StructDeclOp struct_decl)
        -> gap::generator< hl::TypeDeclOp >
    {
        auto module_op = struct_decl->getParentOfType< vast_module >();
        VAST_ASSERT(module_op);

        for (auto decl : top_level_ops< hl::TypeDeclOp >(module_op))
            if (decl.getName() == struct_decl.getName())
                co_yield decl;
    }

    static inline type_generator field_types(mlir::Type t, vast_module module_op)
    {
        auto def = definition_of< hl::StructDeclOp >(t, module_op);
        VAST_CHECK(def, "Was not able to fetch definition of type: {0}", t);
        return field_types(*def);
    }

    hl::ImplicitCastOp implicit_cast_lvalue_to_rvalue(auto &rewriter, auto loc, auto lvalue_op)
    {
        auto lvalue_type = mlir::dyn_cast< hl::LValueType >(lvalue_op.getType());
        VAST_ASSERT(lvalue_type);
        return rewriter.template create< hl::ImplicitCastOp >(
            loc, lvalue_type.getElementType(),
            lvalue_op, hl::CastKind::LValueToRValue);
    }

    // Given record `root` emit `hl::RecordMemberOp` for each its member.
    auto generate_ptrs_to_record_members(operation root, auto loc, auto &bld)
        -> gap::generator< hl::RecordMemberOp >
    {
        auto module_op = root->getParentOfType< vast_module >();
        VAST_ASSERT(module_op);
        auto def = definition_of< hl::StructDeclOp >(root->getResultTypes()[0], module_op);
        VAST_CHECK(def, "Was not able to fetch definition of type from: {0}", *root);

        for (auto field_def : field_defs(*def))
        {
            VAST_ASSERT(root->getNumResults() == 1);
            auto as_val = root->getResult(0);
            // `hl.member` requires type to be an lvalue.
            auto wrap_type = hl::LValueType::get(module_op.getContext(), field_def.getType());
            co_yield bld.template create< hl::RecordMemberOp >(loc,
                                                               wrap_type,
                                                               as_val,
                                                               field_def.getName());
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



    std::optional< std::size_t > field_idx(llvm::StringRef name, auto struct_decl)
    {
        std::size_t out = 0;
        for (auto field_def : field_defs(struct_decl))
        {
            if (field_def.getName() == name)
                return { out };
            ++out;
        }
        return {};
    }
}
