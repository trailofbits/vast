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
#include "vast/CodeGen/Types.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"

namespace vast::cg
{
    struct codegen_driver;
    struct function_processing_lock;

    struct type_conversion_driver {
        type_conversion_driver(codegen_driver &driver);

        core::FunctionType get_function_type(clang::GlobalDecl decl);
        core::FunctionType get_function_type(const function_info_t &info);

        // Convert type into a mlir_type.
        template< bool lvalue = false >
        mlir_type convert_type(qual_type type);

        void update_completed_type(const clang::TagDecl *tag);

        friend struct function_processing_lock;

      private:
        void start_function_processing(const function_info_t *fninfo);
        void finish_function_processing(const function_info_t *fninfo);

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
