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
    struct DataLayoutBlueprint
    {
        bool try_emplace(mlir::Type, const clang::Type *, const clang::ASTContext &actx);

        // [ byte size, bitsize ] - can differ due to alignment.
        using record_t = std::tuple< uint64_t, uint64_t >;

        inline static const auto hasher = [](const mlir::Type &t) {
            return mlir::hash_value(t);
        };
        std::unordered_map< mlir::Type, dl::DLEntry, decltype(hasher) > entries;
    };

    template< typename Stream >
    auto operator<<(Stream &os, const DataLayoutBlueprint &dl) -> decltype(os << "") {
        for (const auto &[ty, sizes] : dl.entries) {
            os << ty << " ";
            const auto &[byte_s, bit_s] = sizes;
            os << llvm::formatv("[ {}, {} ]\n", byte_s, bit_s);
        }
        return os;
    }

    struct TypeConverter
    {
        using AContext = clang::ASTContext;
        using MContext = mlir::MLIRContext;

        TypeConverter(MContext &mctx, AContext &actx) : mctx(mctx), actx(actx) {}

        mlir::Type convert(clang::QualType ty);
        mlir::Type convert(const clang::RecordType *ty);
        mlir::Type convert(const clang::Type *ty, clang::Qualifiers quals);

        mlir::FunctionType convert(const clang::FunctionType *ty);

        // We need to emit data layout - that means we need to remember for
        // converted type its bitsize. For now each conversion functions is *required*
        // to use `dl_aware_convert` which handles the data layout information retrieval.
        // Function that do the conversion itself are private as for now there is no
        // use-case that would require them exposed.
        mlir::Type dl_aware_convert(const clang::Type *ty, clang::Qualifiers quals);

        std::string format_type(const clang::Type *type) const;

        DataLayoutBlueprint take_dl() { return std::move(dl); }

    private:
        mlir::Type do_convert(const clang::Type *ty, clang::Qualifiers quals);
        mlir::Type do_convert(const clang::BuiltinType *ty, clang::Qualifiers quals);
        mlir::Type do_convert(const clang::PointerType *ty, clang::Qualifiers quals);
        mlir::Type do_convert(const clang::RecordType *ty, clang::Qualifiers quals);
        mlir::Type do_convert(const clang::ConstantArrayType *ty, clang::Qualifiers quals);

        MContext &mctx;
        AContext &actx;
        DataLayoutBlueprint dl;
    };

} // namespace vast::hl
