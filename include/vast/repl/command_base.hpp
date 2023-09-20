// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/TypeList.hpp"


namespace vast::repl
{
    struct state_t;

    namespace cmd
    {
        struct base {
            using command_params = util::type_list<>;

            virtual void run(state_t &) const = 0;
            virtual ~base() = default;
        };
    } // namespace cmd

    using command_ptr = std::unique_ptr< cmd::base >;
} // namespace vast::repl
