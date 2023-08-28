// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"

namespace vast::repl {

    struct state_t {
        explicit state_t(mcontext_t &ctx) : ctx(ctx) {}

        bool exit = false;

        std::optional< std::string > source;

        mcontext_t &ctx;
        std::optional< tw::default_tower > tower;
    };

} // namespace vast::repl
