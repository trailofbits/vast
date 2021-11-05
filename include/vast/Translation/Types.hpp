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

namespace vast::hl
{
    struct TypeConverter
    {
        using Context = mlir::MLIRContext;

        TypeConverter(Context *ctx) : ctx(ctx) {}

        mlir::Type convert(clang::QualType ty);

        mlir::Type convert(const clang::Type *ty, clang::Qualifiers quals);
        mlir::Type convert(const clang::BuiltinType *ty, clang::Qualifiers quals);
        mlir::FunctionType convert(const clang::FunctionType *ty);

    private:
        Context *ctx;
    };

} // namespace vast::hl
