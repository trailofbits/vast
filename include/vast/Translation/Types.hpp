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

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::hl
{
    struct TypeConverter
    {
        using MContext = mlir::MLIRContext;
        using ASTContext = clang::ASTContext;

        TypeConverter(MContext *mctx, ASTContext &actx) : mctx(mctx), actx(actx) {}

        mlir::Type convert(clang::QualType ty);

        mlir::Type convert(const clang::Type *ty, clang::Qualifiers quals);
        mlir::Type convert(const clang::BuiltinType *ty, clang::Qualifiers quals);
        mlir::Type convert(const clang::PointerType *ty, clang::Qualifiers quals);
        mlir::Type convert(const clang::RecordType *ty, clang::Qualifiers quals);
        mlir::Type convert(const clang::ArrayType *ty, clang::Qualifiers quals);

        mlir::FunctionType convert(const clang::FunctionType *ty);

        std::string format_type(const clang::Type *type) const;

    private:
        MContext *mctx;
        ASTContext &actx;
    };

} // namespace vast::hl
