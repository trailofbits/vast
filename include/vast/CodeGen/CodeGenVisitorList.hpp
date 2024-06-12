// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include <type_traits>

namespace vast::cg {

    struct visitor_node;

    struct visitor_list {

        visitor_view head_view() const;

        std::shared_ptr< visitor_node > head;
        std::shared_ptr< visitor_node > tail;
    };

    namespace detail {
        template< typename func_t, typename result_t >
        concept builder = std::invocable< func_t, visitor_list >
            && std::convertible_to< std::invoke_result_t< func_t, visitor_list >, result_t >;
    }

    template< typename func_t >
    concept visitor_builder = detail::builder< func_t, std::shared_ptr< visitor_base > >;

    struct visitor_node : std::enable_shared_from_this< visitor_node >, visitor_base
    {
      private:
        struct visitor_node_tag_t {};
        constexpr static visitor_node_tag_t visitor_node_tag{};

        struct deferred_visitor_ctor_tag_t {};
        constexpr static deferred_visitor_ctor_tag_t deferred_visitor_ctor_tag{};

      public:

        visitor_node(visitor_node_tag_t, std::shared_ptr< visitor_base > &&visitor)
            : visitor(std::move(visitor)), next(nullptr)
        {}

        visitor_node(deferred_visitor_ctor_tag_t)
            : visitor(nullptr), next(nullptr)
        {}

        virtual ~visitor_node() = default;

        /**
         * This function template creates a new instance of a visitor_list,
         * which is a shared pointer to a derived_visitor_node.
         * The derived_visitor_node is constructed using the provided arguments args_t.
         * The created node's head is set to itself.
         */
        template< typename derived_visitor_node, typename ...args_t >
        static visitor_list make(args_t &&...args) {
            auto node = std::make_shared< derived_visitor_node >(
                visitor_node_tag, std::forward< args_t >(args)...
            );

            return {node, node};
        }

        /**
         * This function defers the construction of the visitor until the
         * visitor list has a head pointer, as the visitor may use the head
         * pointer to create a head view.
         */
        template< typename derived_visitor_node, visitor_builder builder_t >
        static visitor_list make(builder_t &&builder) {
            auto node = std::make_shared< derived_visitor_node >(deferred_visitor_ctor_tag);

            visitor_list list = {node, node};
            node->visitor = std::invoke(std::forward< builder_t >(builder), list);
            return list;
        }

        std::shared_ptr< visitor_base > visitor;
        std::shared_ptr< visitor_node > next;
    };

    visitor_view visitor_list::head_view() const { return visitor_view(*head); }

    visitor_list operator|(visitor_list lhs, visitor_list &&rhs) {
        lhs.tail->next = std::move(rhs.head);
        lhs.tail       = std::move(rhs.tail);
        return lhs;
    }

    template< typename type >
    visitor_list operator|(visitor_list list, std::optional< type > &&visitor) {
        if (visitor) {
            return list | *visitor;
        } else {
            return list;
        }
    }

    visitor_list operator|(visitor_list node, visitor_builder auto &&visitor_builder) {
       return node | visitor_builder(node);
    }

} // namespace vast::cg
