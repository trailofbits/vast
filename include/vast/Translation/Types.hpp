// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <clang/AST/AST.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Type.h>
VAST_UNRELAX_WARNINGS

#include <tuple>
#include <unordered_map>

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::hl
{
    // For each type remember its data layout information.
    struct DataLayoutBlueprint {
        struct H { auto operator()(const mlir::Type &t) const { return mlir::hash_value(t); } };

        bool try_emplace(mlir::Type, const clang::Type *, const clang::ASTContext &actx);

        // [ byte size, bitsize ] - can differ due to alignment.
        using record_t = std::tuple< uint64_t, uint64_t >;
        std::unordered_map< mlir::Type, dl::DLEntry, H > entries;
    };

    template< typename Stream >
    auto operator<<(Stream &os, const DataLayoutBlueprint &dl) -> decltype(os << "") {
        for (const auto &[ty, sizes] : dl.entries) {
            ty.print(os);
            const auto &[byte_s, bit_s] = sizes;
            os << "[ " << byte_s << ", " << bit_s << " ]\n";
        }
        return os;
    }

    struct TypeConverter
    {
        using AContext = clang::ASTContext;
        using MContext = mlir::MLIRContext;

        TypeConverter(MContext &mctx, AContext &actx) : mctx(mctx), actx(actx) {}

        mlir::Type convert(clang::QualType ty);
        mlir::Type convert(const clang::Type *ty, clang::Qualifiers quals);

        mlir::FunctionType convert(const clang::FunctionType *ty);

        mlir::Type dl_aware_convert(const clang::Type *ty, clang::Qualifiers quals);

        std::string format_type(const clang::Type *type) const;

        DataLayoutBlueprint take_dl() { return std::move(dl); }

    private:
        mlir::Type _convert(const clang::Type *ty, clang::Qualifiers quals);
        mlir::Type _convert(const clang::BuiltinType *ty, clang::Qualifiers quals);
        mlir::Type _convert(const clang::PointerType *ty, clang::Qualifiers quals);
        mlir::Type _convert(const clang::RecordType *ty, clang::Qualifiers quals);
        mlir::Type _convert(const clang::ConstantArrayType *ty, clang::Qualifiers quals);

        MContext &mctx;
        AContext &actx;
        DataLayoutBlueprint dl;
    };

} // namespace vast::hl
