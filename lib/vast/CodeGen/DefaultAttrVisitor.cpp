// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultAttrVisitor.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"

namespace vast::cg
{
    mlir_attr default_attr_visitor::VisitConstAttr(const clang::ConstAttr *attr) {
        return make< hl::ConstAttr >();
    }

    mlir_attr default_attr_visitor::VisitSectionAttr(const clang::SectionAttr *attr) {
        return make< hl::SectionAttr >(attr->getName());
    }

    mlir_attr default_attr_visitor::VisitFormatAttr(const clang::FormatAttr *attr) {
        return make< hl::FormatAttr >(attr->getType()->getName());
    }

    mlir_attr default_attr_visitor::VisitAnnotateAttr(const clang::AnnotateAttr *attr) {
        return make< hl::AnnotationAttr >(attr->getAnnotation());
    }

    mlir_attr default_attr_visitor::VisitAlignedAttr(const clang::AlignedAttr *attr) {
        // TODO(#595): Keep the alignment in the attribute
        return make< hl::AlignedAttr >();
    }

    mlir_attr default_attr_visitor::VisitAlwaysInlineAttr(const clang::AlwaysInlineAttr *) {
        return make< hl::AlwaysInlineAttr >();
    }

    mlir_attr default_attr_visitor::VisitLoaderUninitializedAttr(const clang::LoaderUninitializedAttr *) {
        return make< hl::LoaderUninitializedAttr >();
    }

    mlir_attr default_attr_visitor::VisitNoInstrumentFunctionAttr(const clang::NoInstrumentFunctionAttr *) {
        return make< hl::NoInstrumentFunctionAttr >();
    }

    mlir_attr default_attr_visitor::VisitPackedAttr(const clang::PackedAttr *) {
        return make< hl::PackedAttr >();
    }

    mlir_attr default_attr_visitor::VisitPureAttr(const clang::PureAttr *) {
        return make< hl::PureAttr >();
    }

    mlir_attr default_attr_visitor::VisitWarnUnusedResultAttr(const clang::WarnUnusedResultAttr *) {
        return make< hl::WarnUnusedResultAttr >();
    }

    mlir_attr default_attr_visitor::VisitRestrictAttr(const clang::RestrictAttr *) {
        return make< hl::RestrictAttr >();
    }

    mlir_attr default_attr_visitor::VisitNoThrowAttr(const clang::NoThrowAttr *) {
        return make< hl::NoThrowAttr >();
    }

    mlir_attr default_attr_visitor::VisitNonNullAttr(const clang::NonNullAttr *) {
        return make< hl::NonNullAttr >();
    }

    mlir_attr default_attr_visitor::VisitModeAttr(const clang::ModeAttr *attr) {
        return make< hl::ModeAttr >(attr->getMode()->getName());
    }

    mlir_attr default_attr_visitor::VisitBuiltinAttr(const clang::BuiltinAttr *attr) {
        return make< hl::BuiltinAttr >(attr->getID());
    }

    mlir_attr default_attr_visitor::VisitAsmLabelAttr(const clang::AsmLabelAttr *attr) {
        return make< hl::AsmLabelAttr >(attr->getLabel(), attr->getIsLiteralLabel());
    }

    mlir_attr default_attr_visitor::VisitAllocAlignAttr(const clang::AllocAlignAttr *attr) {
        return make< hl::AllocAlignAttr >(attr->getParamIndex().getSourceIndex());
    }

    mlir_attr default_attr_visitor::VisitAllocSizeAttr(const clang::AllocSizeAttr *attr) {
        auto num = attr->getNumElemsParam().isValid() ? attr->getNumElemsParam().getSourceIndex() : int();
        return make< hl::AllocSizeAttr >(attr->getElemSizeParam().getSourceIndex(), num);
    }

    mlir_attr default_attr_visitor::VisitLeafAttr(const clang::LeafAttr *attr) {
        return make< hl::LeafAttr >();
    }

    mlir_attr default_attr_visitor::VisitColdAttr(const clang::ColdAttr *attr) {
        return make< hl::ColdAttr >();
    }

    mlir_attr default_attr_visitor::VisitDeprecatedAttr(const clang::DeprecatedAttr *attr) {
        return make< hl::DeprecatedAttr >(attr->getMessage(), attr->getReplacement());
    }
} // namespace vast::hcg
