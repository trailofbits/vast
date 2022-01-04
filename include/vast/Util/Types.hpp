// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/APInt.h>
VAST_UNRELAX_WARNINGS

#include <limits>

namespace vast
{
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

} // namespace vast
