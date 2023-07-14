// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Interfaces/SymbolInterface.hpp"

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

    static inline type_generator field_types(hl::StructDeclOp op)
    {
        if (op.getFields().empty())
            co_return;

        for (auto &maybe_field : op.getOps())
        {
            // TODO(hl): So normally only `hl.field` should be present here,
            //           but currently also re-declarations of nested structures
            //           are here - add hard fail if the conversion fails in the future.
            if (mlir::isa< hl::StructDeclOp >(maybe_field))
                continue;

            auto field_decl = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
            VAST_ASSERT(field_decl);
            co_yield field_decl.getType();
        }
    }

    static inline type_generator field_types(mlir::Type t, vast_module mod)
    {
        auto type_name = hl::name_of_record(t);
        VAST_ASSERT(type_name);
        for (auto op : get_nested< hl::StructDeclOp >(mod))
            if (op.getName() == *type_name)
                return field_types(op);
        VAST_UNREACHABLE("Was not able to fetch definition of type: {0}", t);
    }
}
