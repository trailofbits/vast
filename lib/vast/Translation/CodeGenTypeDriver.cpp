

// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Translation/CodeGenTypeDriver.hpp"
#include "vast/Translation/CodeGenDriver.hpp"
#include "vast/Translation/Error.hpp"

#include "vast/CodeGen/ArgInfo.hpp"

namespace vast::cg
{
    type_conversion_driver::type_conversion_driver(codegen_driver &driver)
        : driver(driver)
    {}

    mlir_type type_conversion_driver::convert_type(qual_type /* type */) {
        throw cg::unimplemented("type_conversion_driver::convert_type");
    }

    mlir::FunctionType type_conversion_driver::get_function_type(clang::GlobalDecl /* decl */) {
        throw cg::unimplemented("get_function_type");
    }

    mlir_type type_conversion_driver::get_coerce_to_type(const abi_arg_info &info) {
        throw cg::unimplemented("get_coerce_to_type");
    }

    mlir::FunctionType type_conversion_driver::get_function_type(const function_info_t &fninfo) {
        auto lock = driver.make_lock(&fninfo);
        using abi_kind = abi_arg_info::abi_arg_kind;

        auto process_type_info = [&] (auto info) -> mlir_type {
            switch (info.get_kind()) {
            case abi_kind::ignore:
                // TODO: This should probably be the None type from the builtin
                // dialect.
                return nullptr;
            case abi_kind::extend:
            case abi_kind::direct:
                return get_coerce_to_type(info);
            default:
                throw cg::codegen_error("unsupported abi kind");
            }
        };

        auto rty = process_type_info(fninfo.get_return_info());

        clang_to_vast_arg_mapping vast_function_args(
            driver.acontext(), fninfo, true /* only_required_args */
        );

        llvm::SmallVector< mlir_type , 8> arg_types(vast_function_args.get_total_vast_args());

        assert(!vast_function_args.has_sret_arg() && "NYI");
        assert(!vast_function_args.has_inalloca_arg() && "NYI");

        // Add in all of the required arguments.
        auto end = std::next(fninfo.arg_begin(), fninfo.get_num_required_args());
        unsigned arg_no = 0;
        for (auto it = fninfo.arg_begin(); it != end; ++it, ++arg_no) {
            const auto &arg_info = it->info;
            assert(!vast_function_args.has_padding_arg(arg_no) && "NYI");

            auto [first_vast_arg, num_vast_args] = vast_function_args.get_vast_args(arg_no);
            assert(first_vast_arg == 1);

            arg_types[first_vast_arg] = process_type_info(arg_info);
        }

        auto *mctx = &driver.mcontext();
        return mlir::FunctionType::get(mctx, arg_types, rty ? rty : mlir::TypeRange());
    }

    void type_conversion_driver::start_function_processing(const function_info_t *fninfo) {
        if (bool inserted = functions_being_processed.insert(fninfo).second; !inserted) {
            throw cg::codegen_error("trying to process a function recursively");
        }
    }

    void type_conversion_driver::finish_function_processing(const function_info_t *fninfo) {
        if (auto erased = functions_being_processed.erase(fninfo); !erased) {
            throw cg::codegen_error("function info not being processed");
        }
    }

} // namespace vast::cg
