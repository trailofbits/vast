// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

namespace vast::repl {

    using owning_module_ref = OwningModuleRef;

    struct state_t {
        bool exit = false;

        std::optional< std::string > source;

        owning_module_ref mod;
    };

} // namespace vast::repl
