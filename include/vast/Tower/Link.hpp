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

    using operations = std::vector< operation >;

    // Maybe we want to abstract this as an interface - then it can be lazy as well, for example
    // we could pass in only some step chain and location info and it will get computed.
    // Since if we have `A -> B` we can always construct `B -> A` it may make sense to simply only
    // export bidirectional mapping?
    using op_mapping = llvm::DenseMap< operation, operations >;

    // Generic interface to generalize the connection between any two modules.
    // There are no performance guarantees in general, but there should be implementations
    // available that try to be as performant as possible.
    struct link_interface
    {
        virtual ~link_interface() = default;

        // These are not forced as `const` to allow runtime caching.
        virtual operations children(operation)  = 0;
        virtual operations children(operations) = 0;

        virtual operations parents(operation)  = 0;
        virtual operations parents(operations) = 0;

        virtual op_mapping parents_to_children() = 0;
        virtual op_mapping children_to_parents() = 0;

        virtual handle_t parent() const = 0;
        virtual handle_t child() const = 0;
    };

    namespace views {
        static inline auto parents_to_children = [](const auto &link) {
            return link->parents_to_children();
        };

        static inline auto children_to_parents = [](const auto &link) {
            return link->children_to_parents();
        };

    } // namespace views

    // Represent application of some passes. Invariant is that
    // `parent -> child` are tied by the `location_info`.
    // TODO: How to enforce this - private ctor and provide a builder interface on the side
    //       that is a friend and allowed to create these?
    struct conversion_step : link_interface {
      protected:
        handle_t _parent;
        handle_t _child;
        location_info_t &_location_info;

      public:
        explicit conversion_step(handle_t parent, handle_t child, location_info_t &location_info)
            : _parent(parent), _child(child), _location_info(location_info)
        {}

        operations children(operation) override;
        operations children(operations) override;

        operations parents(operation) override;
        operations parents(operations) override;

        op_mapping parents_to_children() override;
        op_mapping children_to_parents() override;

        handle_t parent() const override;
        handle_t child() const override;
    };

    using link_ptr = std::unique_ptr< link_interface >;
    using link_vector = std::vector< link_ptr >;

    using conversion_steps = std::vector< conversion_step >;

    // `A -> ... -> E` - each middle link is kept and there is pre-computed
    // mapping for `A <-> E` transition to make it more performant.
    struct fat_link : link_interface
    {
      protected:
        link_vector _links;

        op_mapping _to_children;
        op_mapping _to_parents;

      public:
        explicit fat_link(link_vector links);
        fat_link() = delete;

        operations children(operation) override;
        operations children(operations) override;

        operations parents(operation) override;
        operations parents(operations) override;

        op_mapping parents_to_children() override;
        op_mapping children_to_parents() override;

        handle_t child() const override;
        handle_t parent() const override;
    };

} // namespace vast::tw
