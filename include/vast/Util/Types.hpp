// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include <limits>

namespace vast
{
    // TODO(Heno): move to gap
    template< typename F, typename... Args >
    concept invocable = requires(F &&f, Args &&...args) {
        std::invoke(std::forward< F >(f), std::forward< Args >(args)...);
    };

    // TODO(Heno): use integral concepts
    template< typename I >
    constexpr int bits()
    {
        return std::numeric_limits< I >::digits + std::numeric_limits< I >::is_signed;
    }

    template< typename I >
    constexpr auto apint( I value )
    {
        static_assert( bits< I >() <= bits< uint64_t >());
        return llvm::APInt( bits< I >(), uint64_t(value), std::numeric_limits< I >::is_signed );
    }

    template< typename I >
    constexpr auto apsint( I value )
    {
        return llvm::APSInt( apint(value), std::numeric_limits< I >::is_signed );
    }

    static inline std::string format_type(mlir_type type) {
        std::string name;
        llvm::raw_string_ostream os(name);
        type.print(os);
        return name;
    }

} // namespace vast
