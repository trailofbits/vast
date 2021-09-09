// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <mlir/IR/BuiltinTypes.h>
#include <clang/AST/Type.h>
#include <mlir/IR/MLIRContext.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/AST.h>

namespace vast::hl
{
    using qualifiers   = clang::Qualifiers;

    struct TypeConverter
    {
        using context = mlir::MLIRContext;

        TypeConverter(context *ctx) : ctx(ctx) {}

        mlir::Type convert(clang::QualType ty);

        mlir::Type convert(const clang::Type *ty, qualifiers quals);
        mlir::Type convert(const clang::BuiltinType *ty, qualifiers quals);
        mlir::FunctionType convert(const clang::FunctionType *ty);

    private:
        context *ctx;
    };

} // namespace vast::hl
