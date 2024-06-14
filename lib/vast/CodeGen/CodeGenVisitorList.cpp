// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenVisitorList.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    visitor_list_ptr operator|(visitor_list_ptr &&list, visitor_node_ptr &&node) {
        if (!list->head) {
            list->head = node;
            list->tail = std::move(node);
        } else {
            list->tail->next = node;
            list->tail = std::move(node);
        }

        return list;
    }

    visitor_list_ptr operator|(visitor_list_ptr &&list, node_with_list_ref_wrap &&wrap) {
        auto &list_ref = *list;
        return std::move(list) | wrap(list_ref);
    }

} // namespace vast::cg
