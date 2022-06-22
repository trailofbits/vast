// Copyright (c) 2022, Trail of Bits, Inc.
#pragma once

#include <tuple>
#include <utility>

namespace vast::util
{
    namespace detail
    {
        template< std::size_t... Ns, typename... Ts >
        const auto& tail_impl(std::index_sequence< Ns... >, const std::tuple< Ts... >& t) {
            return std::make_tuple(std::get< Ns + 1u >(t)...);
        }
    } // namespace detail

    template< typename T, typename... Ts >
    const auto& head(const std::tuple< T, Ts... >& t) {
        return std::get< 0 >(t);
    }

    template< typename... Ts >
    const auto& tail(const std::tuple< Ts... >& t) {
        return detail::tail_impl(std::make_index_sequence< sizeof...(Ts) - 1u >(), t);
    }
} // namespace vast::util
