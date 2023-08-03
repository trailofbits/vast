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
    gap::generator< T > get_nested(vast_module mod)
    {
        auto body = mod.getBody();
        if (!body)
            co_return;

        for (auto &op : *body)
            if (auto casted = mlir::dyn_cast< T >(op))
                co_yield casted;
    }

    using type_generator = gap::generator< mlir::Type >;
    using value_generator = gap::generator< mlir::Value >;

    static inline gap::generator< hl::FieldDeclOp > field_defs(hl::StructDeclOp op)
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

    static inline type_generator field_types(hl::StructDeclOp op)
    {
        for (auto def : field_defs(op))
            co_yield def.getType();
    }

    static inline auto definition_of(mlir::Type t, vast_module mod)
        -> std::optional< hl::StructDeclOp >
    {
        auto type_name = hl::name_of_record(t);
        VAST_CHECK(type_name, "hl::name_of_record failed with {0}", t);
        for (auto op : get_nested< hl::StructDeclOp >(mod))
            if (op.getName() == *type_name)
                return { op };
        return {};
    }

    static inline type_generator field_types(mlir::Type t, vast_module mod)
    {
        auto def = definition_of(t, mod);
        VAST_CHECK(def, "Was not able to fetch definition of type: {0}", t);
        return field_types(*def);
    }

    // TODO(hl): Custom hook to provide a location?
    auto traverse_record(operation root, auto &bld)
        -> gap::generator< hl::RecordMemberOp >
    {
        auto mod = root->getParentOfType< vast_module >();
        VAST_ASSERT(mod);
        auto def = definition_of(root->getResultTypes()[0], mod);
        VAST_CHECK(def, "Was not able to fetch definition of type from: {0}", *root);

        for (auto field_def : field_defs(*def))
        {
            auto as_val = root->getResult(0);
            // `hl.member` requires type to be an lvalue.
            auto wrap_type = hl::LValueType::get(mod.getContext(), field_def.getType());
            co_yield bld.template create< hl::RecordMemberOp >(root->getLoc(),
                                                               wrap_type,
                                                               as_val,
                                                               field_def.getName());
        }
    }
}
