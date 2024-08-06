// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/ClangVisitorBase.hpp"

namespace vast::cg {

    struct default_attr_visitor : attr_visitor_base< default_attr_visitor >
    {
        using base = attr_visitor_base< default_attr_visitor >;
        using base::base;

        using attr_visitor_base< default_attr_visitor >::Visit;

        mlir_attr visit(const clang_attr *attr) { return Visit(attr); }

        mlir_attr VisitConstAttr(const clang::ConstAttr *attr);
        mlir_attr VisitSectionAttr(const clang::SectionAttr *attr);
        mlir_attr VisitAliasAttr(const clang::AliasAttr *attr);
        mlir_attr VisitFormatAttr(const clang::FormatAttr *attr);
        mlir_attr VisitAnnotateAttr(const clang::AnnotateAttr *attr);
        mlir_attr VisitAlignedAttr(const clang::AlignedAttr *attr);
        mlir_attr VisitAlwaysInlineAttr(const clang::AlwaysInlineAttr *attr);
        mlir_attr VisitNoInlineAttr(const clang::NoInlineAttr *attr);
        mlir_attr VisitLoaderUninitializedAttr(const clang::LoaderUninitializedAttr *attr);
        mlir_attr VisitNoInstrumentFunctionAttr(const clang::NoInstrumentFunctionAttr *attr);
        mlir_attr VisitPackedAttr(const clang::PackedAttr *attr);
        mlir_attr VisitPureAttr(const clang::PureAttr *attr);
        mlir_attr VisitWarnUnusedResultAttr(const clang::WarnUnusedResultAttr *attr);
        mlir_attr VisitRestrictAttr(const clang::RestrictAttr *attr);
        mlir_attr VisitNoThrowAttr(const clang::NoThrowAttr *attr);
        mlir_attr VisitNonNullAttr(const clang::NonNullAttr *attr);
        mlir_attr VisitModeAttr(const clang::ModeAttr *attr);
        mlir_attr VisitBuiltinAttr(const clang::BuiltinAttr *attr);
        mlir_attr VisitAsmLabelAttr(const clang::AsmLabelAttr *attr);
        mlir_attr VisitAllocAlignAttr(const clang::AllocAlignAttr *attr);
        mlir_attr VisitAllocSizeAttr(const clang::AllocSizeAttr *attr);
        mlir_attr VisitLeafAttr(const clang::LeafAttr *attr);
        mlir_attr VisitColdAttr(const clang::ColdAttr *attr);
        mlir_attr VisitDeprecatedAttr(const clang::DeprecatedAttr *attr);
        mlir_attr VisitTransparentUnionAttr(const clang::TransparentUnionAttr *attr);
        mlir_attr VisitReturnsTwiceAttr(const clang::ReturnsTwiceAttr *attr);
        mlir_attr VisitMayAliasAttr(const clang::MayAliasAttr *attr);
        mlir_attr VisitUnusedAttr(const clang::UnusedAttr *attr);
        mlir_attr VisitUsedAttr(const clang::UsedAttr *attr);
        mlir_attr VisitGNUInlineAttr(const clang::GNUInlineAttr *attr);
        mlir_attr VisitAnyX86NoCfCheckAttr(const clang::AnyX86NoCfCheckAttr *attr);
        mlir_attr VisitMaxFieldAlignmentAttr(const clang::MaxFieldAlignmentAttr *attr);
        mlir_attr VisitAvailableOnlyInDefaultEvalMethodAttr(const clang::AvailableOnlyInDefaultEvalMethodAttr *attr);
        mlir_attr VisitAvailabilityAttr(const clang::AvailabilityAttr *attr);

      private:
        template< typename attr_t, typename... args_t >
        auto make(args_t &&...args) {
            return bld.getAttr< attr_t >(std::forward< args_t >(args)...);
        }
    };

} // namespace vast::cg
