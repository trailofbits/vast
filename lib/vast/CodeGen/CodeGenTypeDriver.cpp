

// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenTypeDriver.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"

#include "vast/CodeGen/ArgInfo.hpp"

namespace vast::cg
{
    type_conversion_driver::type_conversion_driver(codegen_driver &driver)
        : driver(driver)
    {}

    template< bool lvalue >
    mlir_type type_conversion_driver::convert_type(qual_type type) {
        mlir_type result = driver.codegen.convert(type);

        if constexpr (lvalue) {
            return driver.codegen.make_lvalue(result);
        }

        return result;
    }

    // UpdateCompletedType - When we find the full definition for a TagDecl,
    // replace the 'opaque' type we previously made for it if applicable.
    void type_conversion_driver::update_completed_type(const clang::TagDecl * /* tag */) {
        // TODO: we probably do not need this
        // this should be resolved as a separate vast pass
        // when we lower enums and structs to ll/core types
    }

    core::FunctionType type_conversion_driver::get_function_type(clang::GlobalDecl /* decl */) {
        VAST_UNIMPLEMENTED;
    }

    core::FunctionType type_conversion_driver::get_function_type(const function_info_t &fninfo) {
        auto lock = driver.make_lock(&fninfo);
        using abi_kind = abi_arg_info::abi_arg_kind;

        auto process_type_info = [&] (const auto &info, auto type, bool arg = true) -> mlir_type {
            switch (info.get_kind()) {
            case abi_kind::ignore:
            case abi_kind::extend:
            case abi_kind::direct:
                if (arg)
                    return convert_type< true >(type);
                return convert_type(type);
            default:
                VAST_UNREACHABLE("unsupported abi kind");
            }
        };

        auto rty = process_type_info(fninfo.get_return_info(), fninfo.get_return_type(), false);

        clang_to_vast_arg_mapping vast_function_args(
            driver.acontext(), fninfo, true /* only_required_args */
        );

        llvm::SmallVector< mlir_type , 8> arg_types(vast_function_args.get_total_vast_args());

        VAST_CHECK(!vast_function_args.has_sret_arg(), "NYI");
        VAST_CHECK(!vast_function_args.has_inalloca_arg(), "NYI");

        // Add in all of the required arguments.
        auto end = std::next(fninfo.arg_begin(), fninfo.get_num_required_args());
        unsigned arg_no = 0;
        for (auto it = fninfo.arg_begin(); it != end; ++it, ++arg_no) {
            const auto &arg_info = it->info;
            VAST_CHECK(!vast_function_args.has_padding_arg(arg_no), "NYI");

            auto [first_vast_arg, num_vast_args] = vast_function_args.get_vast_args(arg_no);
            VAST_ASSERT(num_vast_args == 1);

            arg_types[first_vast_arg] = process_type_info(arg_info, it->type);
        }

        return core::FunctionType::get(arg_types, {rty}, fninfo.is_variadic());
    }

    void type_conversion_driver::start_function_processing(const function_info_t *fninfo) {
        if (bool inserted = functions_being_processed.insert(fninfo).second; !inserted) {
            VAST_UNREACHABLE("trying to process a function recursively");
        }
    }

    void type_conversion_driver::finish_function_processing(const function_info_t *fninfo) {
        if (auto erased = functions_being_processed.erase(fninfo); !erased) {
            VAST_UNREACHABLE("function info not being processed");
        }
    }

} // namespace vast::cg
