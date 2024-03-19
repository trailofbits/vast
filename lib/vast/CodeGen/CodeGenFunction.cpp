// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenFunction.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/Util.hpp"

#include "vast/Dialect/Core/Linkage.hpp"

namespace vast::cg
{
   vast_function function_generator::emit(clang_function *decl) {
        auto &prototype_scope = make_child< prototype_generator >();
        auto prototype = prototype_scope.emit(decl);

        defer([=] {
            auto &body = make_child< body_generator >();
            // TODO pass prototype to body generator
            body.emit(decl);
        });

        return prototype;
    }

    vast_function prototype_generator::emit(clang_function *decl) {
        auto ctx = dynamic_cast< function_context* >(parent);
        VAST_CHECK(ctx, "prototype generator must be a child of a function context");

        auto mod = dynamic_cast< module_context* >(ctx->parent);
        VAST_CHECK(mod, "function context must be a child of a module context");

        auto fty = visitor.visit(decl->getFunctionType(), decl->isVariadic());
        if (auto proto = get_global_value(mod, clang_global(decl))) {
            VAST_ASSERT(mlir::isa< vast_function >(proto));
            VAST_ASSERT(mlir::cast< vast_function >(proto).getFunctionType() == fty);
            return mlir::cast< vast_function >(proto);
        } else {
            auto loc = visitor.location(decl);
            auto mangled_name = get_mangled_name(mod, decl);
            auto linkage = core::get_function_linkage(decl);
            auto visibility = get_function_visibility(decl, linkage);
            auto attrs = get_function_attrs(decl);

            return declare(
                loc, mangled_name, fty, linkage, visibility, std::move(attrs)
            );
        }
    }

    vast_function prototype_generator::declare(
        loc_t loc,
        mangled_name_ref mangled_name,
        mlir_type ty,
        linkage_kind linkage,
        mlir_visibility visibility,
        mlir_attr_list attrs
    ) {
        // At the point we need to create the function, the insertion point
        // could be anywhere (e.g. callsite). Do not rely on whatever it might
        // be, properly save, find the appropriate place and restore.
        auto guard = visitor.insertion_guard();

        auto fty = mlir::cast< core::FunctionType >(ty);
        auto fn = visitor.builder().create< hl::FuncOp >(
            loc, mangled_name.name, fty, linkage
        );

        attrs.append(fn->getAttrs());
        fn->setAttrs(attrs);

        mlir::SymbolTable::setSymbolVisibility(fn, visibility);

        scope_context::declare(fn);
        return fn;
    }

    mlir_attr_list prototype_generator::get_function_attrs(clang_function *decl) {
        if (!decl->hasAttrs())
            return {};

        // These are already handled by linkage attributes
        using excluded_attr_list = util::type_list<
              clang::WeakAttr
            , clang::SelectAnyAttr
            , clang::CUDAGlobalAttr
        >;

        mlir_attr_list attrs;
        for (auto attr : exclude_attrs< excluded_attr_list >(decl->getAttrs())) {
            auto visited = visitor.visit(attr);

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


    mlir_visibility prototype_generator::get_function_visibility(
        clang_function *decl, linkage_kind linkage
    ) {
        if (decl->isThisDeclarationADefinition())
            return core::get_visibility_from_linkage(linkage);
        if (decl->doesDeclarationForceExternallyVisibleDefinition())
            return mlir_visibility::Public;
        return mlir_visibility::Private;
    }

    void body_generator::emit(clang_function *decl) {
        emit_epilogue(decl);
    }

    void body_generator::emit_epilogue(clang_function *decl) {}

} // namespace vast::cg
