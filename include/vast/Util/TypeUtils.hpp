// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <vast/Util/Functions.hpp>
#include <vast/Util/Common.hpp>

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

        mlir::AttrTypeWalker walker;
        walker.addWalk([&](mlir_type t) { found |= accept(t); });

        walker.walk(root);
        return found;
    }

    template< typename ... Ts >
    bool contains_subtype(mlir_type root)
    {
        return contains_subtype(root, [](auto t) { return mlir::isa< Ts ... >(t); });
    }

    // `mlir::TypeRange` does not expose a lot of stuff to write a concept with.
    bool contains_subtype(mlir::TypeRange type_range, auto &&accept)
    {
        for (auto t : type_range)
            if (contains_subtype(t, accept))
                return true;
        return false;
    }

    bool contains_subtype(mlir::DataLayoutEntryInterface dl_entry, auto &&accept)
    {
        auto key = mlir::dyn_cast< mlir_type >(dl_entry.getKey());
        return key && accept(key);
    }

    bool contains_subtype(mlir::Attribute root, auto &&accept)
    {
        bool found = false;
        mlir::AttrTypeWalker walker;
        walker.addWalk([&](mlir::Attribute attr) -> mlir::WalkResult
        {
            if (auto dl_entry = mlir::dyn_cast< mlir::DataLayoutSpecInterface >(attr))
            {
                for (auto e : dl_entry.getEntries())
                    found |= contains_subtype(e, accept);
            }
            return mlir::WalkResult::advance();
        });

        walker.addWalk([&](mlir_type t)
        {
            found |= accept(t);
        });

        walker.walk(root);

        return found;
    }

    // Query whether a type is hidden somewhere - probably useful for type conversions
    // passes.
    bool has_type_somewhere(operation op, auto &&accept)
    {
        auto contains_in_function_type = [&] {
            if (auto fn = mlir::dyn_cast< core::function_op_interface >(op)) {
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

    template< typename user_filter, typename yield_t >
    walk_result type_users(user_filter &&is_user, auto scope, yield_t &&yield) {
        return scope.walk([&](operation op) {
            return has_type_somewhere(op, std::forward< user_filter >(is_user))
                 ? yield(op) : walk_result::advance();
        });
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
