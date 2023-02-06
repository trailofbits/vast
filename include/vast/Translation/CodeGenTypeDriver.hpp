// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/CodeGen/CGFunctionInfo.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/BuiltinTypes.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/FunctionInfo.hpp"
#include "vast/Translation/Types.hpp"

namespace vast::cg
{
    struct codegen_driver;
    struct function_processing_lock;

    struct type_conversion_driver {
        type_conversion_driver(codegen_driver &driver);

        mlir::FunctionType get_function_type(clang::GlobalDecl decl);
        mlir::FunctionType get_function_type(const function_info_t &info);

        // Convert type into a mlir_type.
        mlir_type convert_type(qual_type type);

        mlir_type get_coerce_to_type(const abi_arg_info &info);

        friend struct function_processing_lock;

      private:
        void start_function_processing(const function_info_t *fninfo);
        void finish_function_processing(const function_info_t *fninfo);

        using type_cache_t = llvm::DenseMap< const clang::Type *, mlir_type >;
        type_cache_t type_cache;

        codegen_driver &driver;

        llvm::SmallPtrSet< const function_info_t *, 4 > functions_being_processed;
    };

    struct function_processing_lock {
        function_processing_lock(type_conversion_driver &driver, const function_info_t *fninfo)
            : fninfo(fninfo)
            , driver(driver)
        {
            driver.start_function_processing(fninfo);
        }

        ~function_processing_lock() { driver.finish_function_processing(fninfo); }

      private:
        const function_info_t *fninfo;
        type_conversion_driver &driver;
    };

} // namespace vast::cg
