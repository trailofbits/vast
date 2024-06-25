// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/Tower/Handle.hpp"
#include "vast/Tower/LocationInfo.hpp"

VAST_RELAX_WARNINGS
VAST_UNRELAX_WARNINGS

#include <memory>
#include <vector>

namespace vast::tw {

    struct one_step_link_interface
    {
        virtual ~one_step_link_interface() = default;

        virtual handle_t from() const = 0;
        virtual handle_t to() const   = 0;

        virtual location_info &li() const = 0;
    };

    struct light_one_step_link : one_step_link_interface
    {
      protected:
        handle_t _from;
        handle_t _to;

        location_info &_li;

      public:
        explicit light_one_step_link(handle_t _from, handle_t _to, location_info &_li)
            : _from(_from), _to(_to), _li(_li) {}

        handle_t from() const override { return _from; }

        handle_t to() const override { return _to; }

        location_info &li() const override { return _li; }
    };

    using operations = std::vector< operation >;

    // TODO: Implement.
    struct link_interface
    {
        virtual ~link_interface() = default;

        // These are not forced as `const` to allow runtime caching.
        virtual operations children(operation)  = 0;
        virtual operations children(operations) = 0;

        virtual operations shared_children(operations) = 0;

        virtual operations parents(operation)  = 0;
        virtual operations parents(operations) = 0;

        virtual operations shared_parents(operations) = 0;
    };

    using link_ptr = std::unique_ptr< link_interface >;

    using unit_link_ptr    = std::unique_ptr< one_step_link_interface >;
    using unit_link_vector = std::vector< unit_link_ptr >;

    // This is technically an impl detail but may be handy at multiple placex.
    using op_mapping = std::unordered_map< operation, operations >;

    // `A -> ... -> E` - each middle link is kept and there pre-computed
    // mapping for `A -> E` transition.
    struct fat_link : link_interface
    {
      protected:
        unit_link_vector steps;

        op_mapping down;
        op_mapping up;

      public:
        explicit fat_link(unit_link_vector steps);

        operations children(operation) override;
        operations children(operations) override;

        operations shared_children(operations) override;

        operations parents(operation) override;
        operations parents(operations) override;

        operations shared_parents(operations) override;
    };

} // namespace vast::tw
