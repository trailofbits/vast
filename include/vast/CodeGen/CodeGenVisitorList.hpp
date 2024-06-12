// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include <type_traits>

namespace vast::cg {

    struct visitor_list
    {
        struct node;
        using node_ptr = std::shared_ptr< node >;

        struct node : visitor_base
        {
          protected:
            struct node_tag_t {};

            constexpr static node_tag_t node_tag{};

          public:
            node(node_tag_t) : next(nullptr) {}

            virtual ~node() = default;
            node_ptr next;
        };

        node_ptr head;
        node_ptr tail;
    };

} // namespace vast::cg
