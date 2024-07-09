// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include <gap/coro/generator.hpp>

namespace vast::cg
{
    template< typename T >
    gap::generator< T * > filter(auto from) {
        for (auto x : from) {
            if (auto s = dyn_cast< T >(x))
                co_yield s;
        }
    }
} // namespace vast::cg
