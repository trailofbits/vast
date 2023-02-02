

// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Translation/CodeGenTypeDriver.hpp"
#include "vast/Translation/Error.hpp"

namespace vast::cg
{
    type_conversion_driver::type_conversion_driver(codegen_driver &driver)
        : driver(driver)
    {}

    mlir_type type_conversion_driver::convert_type(qual_type /* type */) {
        throw cg::unimplemented("types_generator::convert_type");
        (void)driver;
    }

    mlir::FunctionType type_conversion_driver::get_function_type(clang::GlobalDecl /* decl */) {
        throw cg::unimplemented("get_function_type");
    }

    mlir::FunctionType type_conversion_driver::get_function_type(const function_info_t & /* info */) {
        throw cg::unimplemented("get_function_type");
    }


} // namespace vast::cg
