// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultDeclVisitor.hpp"

#include "vast/Dialect/Core/Linkage.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    bool unsupported(const clang_function *decl) {
        if (decl->getAttr< clang::ConstructorAttr >()) {
            return true;
        }

        if (decl->getAttr< clang::DestructorAttr >()) {
            return true;
        }

        if (decl->isMultiVersion()) {
            return true;
        }

        if (llvm::dyn_cast< clang::CXXMethodDecl >(decl)) {
            return true;
        }

        return false;
    }

    mlir_visibility get_function_visibility(const clang_function *decl, linkage_kind linkage) {
        if (decl->isThisDeclarationADefinition()) {
            return core::get_visibility_from_linkage(linkage);
        }
        if (decl->doesDeclarationForceExternallyVisibleDefinition()) {
            return mlir_visibility::Public;
        }
        return mlir_visibility::Private;
    }

    operation default_decl_visitor::visit_prototype(const clang_function *decl) {
        if (unsupported(decl)) {
            return {};
        }

        auto set_visibility = [&] (vast_function fn) {
            auto visibility = get_function_visibility(decl, fn.getLinkage());
            mlir::SymbolTable::setSymbolVisibility(fn, visibility);
            return fn;
        };

        auto set_attrs = [&] (vast_function fn) {
            auto attrs = visit_attrs(decl);
            attrs.append(fn->getAttrs());
            fn->setAttrs(attrs);
            return fn;
        };

        return bld.compose< vast_function >()
            .bind(self.location(decl))
            .bind(self.symbol(decl))
            .bind_dyn_cast< vast_function_type >(self.visit(decl->getFunctionType(), decl->isVariadic()))
            .bind(core::get_function_linkage(decl))
            .freeze_as_maybe() // construct vast_function
            .transform(set_visibility)
            .transform(set_attrs)
            .take();
    }

    mlir_attr_list default_decl_visitor::visit_attrs(const clang_function *decl) {
        if (!decl->hasAttrs()) {
            return {};
        }

        // These are already handled by linkage attributes
        using excluded_attr_list = util::type_list<
              clang::WeakAttr
            , clang::SelectAnyAttr
            , clang::CUDAGlobalAttr
        >;

        mlir_attr_list attrs;
        for (auto attr : exclude_attrs< excluded_attr_list >(decl->getAttrs())) {
            auto visited = self.visit(attr);

            auto spelling = attr->getSpelling();
            // Bultin attr doesn't have spelling because it can not be written in code
            if (auto builtin = clang::dyn_cast< clang::BuiltinAttr >(attr)) {
                spelling = "builtin";
            }

            if (auto prev = attrs.getNamed(spelling)) {
                VAST_CHECK(visited == prev.value().getValue(), "Conflicting redefinition of attribute {0}", spelling);
            }

            attrs.set(spelling, visited);
        }

        return attrs;
    }

} // namespace vast::hl
