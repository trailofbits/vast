// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

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
} // namespace vast
