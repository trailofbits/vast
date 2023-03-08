// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include <gap/core/generator.hpp>

#include <functional>
#include <queue>

namespace vast::cg
{
    struct scope_context {
        using action_t = std::function< void() >;

        ~scope_context() {
            for (const auto &action : deferred()) {
                action();
            }

            VAST_ASSERT(deferred_codegen_actions.empty());
        }

        gap::generator< action_t > deferred() {
            while (!deferred_codegen_actions.empty()) {
                co_yield deferred_codegen_actions.front();
                deferred_codegen_actions.pop();
            }
        }

        void defer(action_t action) {
            deferred_codegen_actions.emplace(std::move(action));
        }

        std::queue< action_t > deferred_codegen_actions;
    };

} // namespace vast::cg
