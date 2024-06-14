// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/AttrVisitorProxy.hpp"
#include "vast/CodeGen/Util.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"

#include <ranges>
#include <functional>

namespace vast::cg
{
    operation attr_visitor_proxy::visit_decl_attrs(operation op, const clang_decl *decl, scope_context &scope) {
        if (!decl->hasAttrs()) {
            return op;
        }

        mlir_attr_list attrs = op->getAttrs();

        auto filtered_attrs = decl->getAttrs() | std::ranges::views::filter([&] (auto attr) {
            return !util::is_one_of< excluded_attr_list >(attr);
        });

        for (auto attr : filtered_attrs) {
            auto visited = head.visit(attr, scope);
            auto is_unsup = mlir::isa< unsup::UnsupportedDialect >(visited.getDialect());
            auto key = is_unsup ? attr->getSpelling() : visited.getAbstractAttribute().getName();

            attrs.set(key, visited);
        }

        op->setAttrs(attrs);

        return op;
    }


    operation attr_visitor_proxy::visit(const clang_decl *decl, scope_context &scope) {
        if (auto op = next->visit(decl, scope)) {
            return visit_decl_attrs(op, decl, scope);
        }

        return {};
    }

} // namespace vast::cg
