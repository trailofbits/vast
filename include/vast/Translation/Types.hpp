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
        using Context = mlir::MLIRContext;

        TypeConverter(Context *ctx) : ctx(ctx) {}

        HighLevelType convert(clang::QualType ty);

        HighLevelType convert(const clang::Type *ty, clang::Qualifiers quals);
        HighLevelType convert(const clang::BuiltinType *ty, clang::Qualifiers quals);
        HighLevelType convert(const clang::PointerType *ty, clang::Qualifiers quals);
        mlir::FunctionType convert(const clang::FunctionType *ty);

    private:
        Context *ctx;
    };

} // namespace vast::hl
