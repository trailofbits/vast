// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/AttrVisitor.h>
#include <clang/AST/Attr.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendDiagnostic.h>
VAST_UNRELAX_WARNINGS

#include <mlir/IR/Attributes.h>
#include <vast/CodeGen/Types.hpp>

namespace vast::cg {

    template< typename Derived >
    struct CodeGenAttrVisitor
        : clang::ConstAttrVisitor< CodeGenAttrVisitor< Derived >, operation >
        , CodeGenVisitorLens< CodeGenAttrVisitor< Derived >, Derived >
        , CodeGenBuilder< CodeGenAttrVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenAttrVisitor< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::acontext;

        using LensType::visit;

        using LensType::meta_location;

        using Builder = CodeGenBuilder< CodeGenAttrVisitor< Derived >, Derived >;

        using Builder::builder;

        template< typename Attr, typename... Args >
        auto make(Args &&...args) {
            return builder().template getAttr< Attr >(std::forward< Args >(args)...);
        }

        auto Visit(const clang::Attr *attr) -> mlir_attr {
            if (auto section = mlir::dyn_cast< clang::SectionAttr >(attr)) {
                return VisitSectionAttr(section);
            }
            if (auto annot = mlir::dyn_cast< clang::AnnotateAttr >(attr)) {
                return VisitAnnotateAttr(annot);
            }
            return {};
        }

        hl::SectionAttr VisitSectionAttr(const clang::SectionAttr *attr) {
            std::string name(attr->getName());
            return make< hl::SectionAttr >(name);
        }

        hl::AnnotationAttr VisitAnnotateAttr(const clang::AnnotateAttr *attr) {
            return make< hl::AnnotationAttr >(attr->getAnnotation());
        }

    };
} // namespace vast::cg
