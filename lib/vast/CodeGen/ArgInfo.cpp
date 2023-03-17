// Copyright (c) 2023, Trail of Bits, Inc.

#include <vast/CodeGen/ArgInfo.hpp>
#include <vast/CodeGen/ABIInfo.hpp>

namespace vast::cg
{
    using abi_kind = abi_arg_info::abi_arg_kind;

    void clang_to_vast_arg_mapping::construct(
        const acontext_t &Context, const function_info_t &fninfo, bool only_required_args
    ) {
        unsigned vast_arg_no     = 0;
        bool swap_this_with_sret = false;
        const auto &ret_info = fninfo.get_return_info();

        if (ret_info.get_kind() == abi_kind::indirect) {
            VAST_UNIMPLEMENTED_MSG("info for abi_arg_info::indirect");
        }

        unsigned arg_no   = 0;
        unsigned num_args = only_required_args ? fninfo.get_num_required_args() : fninfo.arg_size();
        for (auto it = fninfo.arg_begin(); arg_no < num_args; ++it, ++arg_no) {
            VAST_ASSERT(it != fninfo.arg_end());
            const auto &ai = it->info;
            // Collect data about vast arguments corresponding to Clang argument arg_no.
            auto &vast_args = arg_info[arg_no];

            // if (ai.get_padding_type()) {
            //     VAST_UNIMPLEMENTED_MSG("padding types info");
            // }

            vast_args.number_of_args = [&] () -> unsigned {
                switch (ai.get_kind()) {
                    case abi_kind::extend:
                    case abi_kind::direct: {
                        // FIXME: deal with struct types
                        return 1;
                    }
                    default: VAST_UNREACHABLE("unsupported abi kind");
                }
            } ();

            if (vast_args.number_of_args > 0) {
                vast_args.first_arg_index = vast_arg_no;
                vast_arg_no += vast_args.number_of_args;
            }

            if (swap_this_with_sret) {
                VAST_UNIMPLEMENTED_MSG("unsupported swap this with sret");
            }
        }
        VAST_ASSERT(arg_no == arg_info.size());

        // VAST_CHECK(!fninfo.uses_in_alloca(), "NYI");

        total_vast_args = vast_arg_no;
    }
} // namespace vast::cg
