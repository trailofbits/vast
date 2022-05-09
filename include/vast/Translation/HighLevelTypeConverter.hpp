// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Type.h>
#include <mlir/IR/BuiltinTypes.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Translation/Context.hpp"
#include "vast/Util/DataLayout.hpp"

#include <tuple>
#include <unordered_map>

namespace vast::hl
{
    using Quals = clang::Qualifiers;

    struct HighLevelTypeConverter {
        HighLevelTypeConverter(TranslationContext &ctx)
            : ctx(ctx) {}

        mlir::Type convert(clang::QualType ty);
        mlir::Type convert(const clang::Type *ty, Quals quals = {});

        mlir::Type lvalue_convert(clang::QualType ty);
        mlir::Type lvalue_convert(const clang::Type *ty, Quals quals = {});

        mlir::FunctionType convert(const clang::FunctionType *ty);

        // We need to emit data layout - that means we need to remember for
        // converted type its bitsize. For now each conversion functions is *required*
        // to use `dl_aware_convert` which handles the data layout information retrieval.
        // Function that do the conversion itself are private as for now there is no
        // use-case that would require them exposed.
        mlir::Type dl_aware_convert(const clang::Type *ty, Quals quals);
        mlir::Type dl_aware_lvalue_convert(const clang::Type *ty, Quals quals);

        std::string format_type(const clang::Type *type) const;

      private:
        mlir::Type do_convert(const clang::Type *ty, Quals quals);
        mlir::Type do_convert(const clang::BuiltinType *ty, Quals quals);
        mlir::Type do_convert(const clang::PointerType *ty, Quals quals);
        mlir::Type do_convert(const clang::ArrayType *ty, Quals quals);

        mlir::Type do_convert(const clang::RecordType *ty, Quals quals);
        mlir::Type do_convert(const clang::EnumType *ty, Quals quals);

        mlir::Type do_convert(const clang::TypedefType *ty, Quals quals);

        TranslationContext &ctx;
    };

} // namespace vast::hl
