// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <utility>
namespace vast
{
    static auto inline identity = [] (auto &&v) {
        return std::forward< decltype(v) >( v );
    };

    template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
    template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

} // namespace vast
