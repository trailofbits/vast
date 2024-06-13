// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"
#include "vast/Tower/Handle.hpp"

VAST_RELAX_WARNINGS
VAST_UNRELAX_WARNINGS

namespace vast::tw {

    struct one_step_link_t {
        handle_t from;
        handle_t to;
    };

    using operations = std::vector< operation >;

    // TODO: Implement.
    struct link_t {
        virtual ~link_t() = default;

        // These are not forced as `const` to allow runtime caching.
        virtual operations children(operation) = 0;
        virtual operations children(operations) = 0;

        virtual operations shared_children(operations) = 0;

        virtual operations parents(operation) = 0;
        virtual operations parents(operations) = 0;

        virtual operations shared_parents(operations) = 0;
    };

} // namespace vast::tw
