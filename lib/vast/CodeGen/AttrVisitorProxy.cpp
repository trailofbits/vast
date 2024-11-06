// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/AttrVisitorProxy.hpp"

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

        llvm::errs() << "visiting attrs\n";
        auto pp = clang::PrintingPolicy(clang::LangOptions());
        for (auto &attr : decl->getAttrs()) {
            if (llvm::isa< clang::BuiltinAttr >(attr)) {
                auto fndecl = llvm::cast< clang::FunctionDecl >(decl);
                llvm::errs() << "builtin attr. does it have id? " << fndecl->getBuiltinID();
            }
            attr->printPretty(llvm::errs(), pp);
            llvm::errs() << "\n";
        }
        auto filtered_attrs = decl->getAttrs() | std::ranges::views::filter([&] (auto attr) {
            return !util::is_one_of< excluded_attr_list >(attr);
        });

        for (auto attr : filtered_attrs) {
            if (auto visited = head.visit(attr, scope)) {
                // Using `attrs.set` instead of `push_back` to ensure that if the
                // attribute already exists, it is updated with the new value. This
                // is crucial for handling cases of redeclaration with a different
                // attribute value.
                //
                // FIXME: This is a temporary solution. We need to handle union
                // of values for the same attribute.
                attrs.set(visited->getName(), visited->getValue());
            }
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
