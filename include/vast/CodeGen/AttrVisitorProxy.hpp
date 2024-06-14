// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorList.hpp"

namespace vast::cg {

    struct attr_visitor_proxy : fallthrough_list_node {

        explicit attr_visitor_proxy(visitor_base &head) : head(head) {}

        using excluded_attr_list = util::type_list<
              clang::WeakAttr
            , clang::SelectAnyAttr
            , clang::CUDAGlobalAttr
        >;

        operation visit_decl_attrs(operation op, const clang_decl *decl, scope_context &scope);

        operation visit(const clang_decl *decl, scope_context &scope) override;

      protected:
        visitor_view head;
    };

} // namespace vast::cg
