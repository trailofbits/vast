// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"
#include "vast/repl/command_base.hpp"

#include <filesystem>

namespace vast::repl {

    struct state_t {
        explicit state_t(mcontext_t &ctx) : ctx(ctx) {}

        bool exit = false;

        std::optional< std::filesystem::path > source;

        mcontext_t &ctx;
        std::optional< tw::default_tower > tower;

        std::vector< command_ptr > sticked;
    };

} // namespace vast::repl
