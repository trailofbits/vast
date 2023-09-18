// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <vast/Util/Functions.hpp>

namespace vast
{
    template< template < typename T > class trait, typename... types >
    bool any_with_trait(types... type)
    {
        return (... || type. template hasTrait< trait >());
    }

    template< template < typename T > class trait, typename... types >
    bool all_with_trait(types... type)
    {
        return (... && type. template hasTrait< trait >());
    }

    bool contains_subtype(mlir_type root, auto &&accept)
    {
        bool found = false;
        auto visitor = [&](auto t) { found |= accept(t); };

        if (auto with_sub_elements = mlir::dyn_cast< mlir::SubElementTypeInterface >(root))
            with_sub_elements.walkSubTypes(visitor);
        visitor(root);
        return found;
    }

    template< typename ... Ts >
    bool contains_subtype(mlir_type root)
    {
        return contains_subtype(root, [](auto t) { return mlir::isa< Ts ... >(t); });
    }

    // `mlir::TypeRange` does not expose a lot of stuff to write a concept with.
    auto contains_subtype(mlir::TypeRange type_range, auto &&accept)
    {
        for (auto t : type_range)
            if (contains_subtype(t, accept))
                return true;
        return false;
    }

    auto contains_subtype(mlir::Attribute root, auto &&accept)
    {
        bool found = false;
        auto visitor = [&](mlir::Attribute attr)
        {
            if (auto type_attr = mlir::dyn_cast< mlir::TypeAttr >(attr))
                found |= contains_subtype(type_attr.getValue(), accept);
            if (auto typed_attr = mlir::dyn_cast< mlir::TypedAttr >(attr))
                found |= contains_subtype(typed_attr.getType(), accept);
        };

        if (auto with_sub_elements = mlir::dyn_cast< mlir::SubElementAttrInterface >(root))
            with_sub_elements.walkSubAttrs(visitor);
        visitor(root);

        return found;
    }

    // Query whether a type is hidden somewhere - probably useful for type conversions
    // passes.
    bool has_type_somewhere(operation op, auto &&accept)
    {
        auto contains_in_function_type = [&] {
            if (auto fn = mlir::dyn_cast< mlir::FunctionOpInterface >(op)) {
                return contains_subtype(fn.getResultTypes(), accept)
                    || contains_subtype(fn.getArgumentTypes(), accept);
            }
            return false;
        };

        return contains_subtype(op->getResultTypes(), accept)
            || contains_subtype(op->getOperandTypes(), accept)
            || contains_subtype(op->getAttrDictionary(), accept)
            || contains_in_function_type();
    }

    template< typename ... Ts >
    bool has_type_somewhere(operation op)
    {
        auto accept = [](auto t) { return mlir::isa< Ts ... >(t); };
        return has_type_somewhere(op, accept);
    }

    auto bw(const auto &dl, mlir_type type) { return dl.getTypeSizeInBits(type); }

    auto bw(const auto &dl, auto type_range)
    {
        std::size_t acc = 0;
        for (auto type : type_range)
            acc += bw(dl, type);
        return acc;
    }
} // namespace vast
