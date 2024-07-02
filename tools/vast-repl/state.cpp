// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/repl/state.hpp"

VAST_RELAX_WARNINGS
VAST_UNRELAX_WARNINGS

namespace vast::repl {

    void state_t::raise_tower(owning_module_ref mod) {
        tower.emplace(ctx, location_info, std::move(mod));
    }

    vast_module state_t::current_module() {
        return tower->top().mod;
    }
} // namespace vast::repl::codegen
