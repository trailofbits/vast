// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/CodeGen/CGFunctionInfo.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/BuiltinTypes.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/Types.hpp"

#include "vast/CodeGen/FunctionInfo.hpp"

namespace vast::cg
{
    struct codegen_driver;

    struct type_conversion_driver {
        type_conversion_driver(codegen_driver &driver);

        mlir::FunctionType get_function_type(clang::GlobalDecl decl);
        mlir::FunctionType get_function_type(const function_info_t &info);

        // Convert type into a mlir_type.
        mlir_type convert_type(qual_type type);

      private:

        using type_cache_t = llvm::DenseMap< const clang::Type *, mlir_type >;
        type_cache_t type_cache;

        codegen_driver &driver;
    };

} // namespace vast::cg
