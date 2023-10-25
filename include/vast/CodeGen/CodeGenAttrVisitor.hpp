// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendDiagnostic.h>
VAST_UNRELAX_WARNINGS

#include <mlir/IR/Attributes.h>
#include <vast/CodeGen/Types.hpp>
#include <vast/CodeGen/CodeGenVisitorLens.hpp>
#include <vast/CodeGen/CodeGenBuilder.hpp>

namespace vast::cg {

    template< typename derived_t >
    struct default_attr_visitor
        : attr_visitor_base< default_attr_visitor< derived_t > >
        , visitor_lens< derived_t, default_attr_visitor >
    {
        using lens = visitor_lens< derived_t, default_attr_visitor >;

        using lens::derived;
        using lens::context;
        using lens::mcontext;
        using lens::acontext;

        using lens::mlir_builder;

        using lens::visit;

        template< typename attr_t, typename... args_t >
        auto make(args_t &&...args) {
            return mlir_builder().template getAttr< attr_t >(
                std::forward< args_t >(args)...
            );
        }

        mlir_attr VisitSectionAttr(const clang::SectionAttr *attr) {
            return make< hl::SectionAttr >(attr->getName());
        }

        mlir_attr VisitAnnotateAttr(const clang::AnnotateAttr *attr) {
            return make< hl::AnnotationAttr >(attr->getAnnotation());
        }

        mlir_attr VisitLoaderUninitializedAttr(const clang::LoaderUninitializedAttr *attr) {
            return make< hl::LoaderUninitializedAttr >();
        }

        mlir_attr VisitNoInstrumentFunctionAttr(const clang::NoInstrumentFunctionAttr *attr) {
            return make< hl::NoInstrumentFunctionAttr >();
        }

        mlir_attr VisitPackedAttr(const clang::PackedAttr *attr) {
            return make< hl::PackedAttr >();
        }

        mlir_attr VisitWarnUnusedResultAttr(const clang::WarnUnusedResultAttr *attr) {
            return make< hl::WarnUnusedResultAttr >();
        }

        mlir_attr VisitRestrictAttr(const clang::RestrictAttr *attr) {
            return make< hl::RestrictAttr >();
        }

        mlir_attr VisitNoThrowAttr(const clang::NoThrowAttr *attr) {
            return make< hl::NoThrowAttr >();
        }

        mlir_attr VisitBuiltinAttr(const clang::BuiltinAttr *attr) {
            return make< hl::BuiltinAttr >(attr->getID());
        }

        mlir_attr VisitAllocSizeAttr(const clang::AllocSizeAttr *attr) {
            auto num = attr->getNumElemsParam().isValid() ? attr->getNumElemsParam().getSourceIndex() : int();
            return make< hl::AllocSizeAttr >(attr->getElemSizeParam().getSourceIndex(), num);
        }
    };
} // namespace vast::cg
