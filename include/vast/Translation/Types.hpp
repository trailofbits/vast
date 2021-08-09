// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <mlir/IR/BuiltinTypes.h>
#include <clang/AST/Type.h>
#include <mlir/IR/MLIRContext.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/AST.h>

namespace vast::hl
{
    struct TypeConverter
    {
        using context = mlir::MLIRContext;

        TypeConverter(context &ctx, clang::ASTContext &ast)
            : ctx(ctx)/*, ast(ast)*/
        {}

        mlir::Type convert(const clang::BuiltinType *ty);

    private:
        context &ctx;
        //clang::ASTContext &ast;
    };

} // namespace vast::hl