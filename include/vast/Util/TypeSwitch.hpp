// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/TypeList.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
VAST_UNRELAX_WARNINGS


namespace vast
{
    // Simple augmentation of `llvm::TypeSwitch` that adds overload of `Case` that
    // accepts type list, which allows us easier dispatch on mlir types that very often
    // do not share predecessor classes.
    template< typename T, typename R >
    struct TypeSwitch : llvm::TypeSwitch< T, R >
    {
        using self_t = TypeSwitch< T, R >;
        using parent_t = llvm::TypeSwitch< T, R >;
        using parent_t::parent_t;

        using parent_t::Default;

        template< typename Callable, typename ... Ts >
        self_t &Case(util::type_list< Ts ... >, Callable &&fn)
        {
            return this->Case< Ts ... >(std::forward< Callable >(fn));
        }

        // This wrapper is required for the return type to be correct.
        template< typename ... Us, typename Callable >
        self_t &Case(Callable &&fn)
        {
            this->parent_t::template Case< Us ... >(std::forward< Callable >(fn));
            return *this;
        }
    };

} // namespace vast
