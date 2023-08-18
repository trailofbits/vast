// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/SmallVector.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/FunctionInfo.hpp"

#include "vast/Util/Common.hpp"


namespace vast::cg
{

    // Encapsulates information about the way function arguments should be
    // passed to actual vast function.
    struct clang_to_vast_arg_mapping {
      private:
        static constexpr unsigned invalid_index = ~0U;
        unsigned inalloca_arg_no;
        unsigned sret_arg_no;
        unsigned total_vast_args;

        // Arguments of vast function corresponding to single clang argument.
        struct vast_args {
            unsigned padding_arg_index = 0;
            // Argument is expanded to vast arguments at positions
            // [first_arg_index, first_arg_index + number_of_args).
            unsigned first_arg_index = 0;
            unsigned number_of_args  = 0;

            vast_args()
                : padding_arg_index(invalid_index)
                , first_arg_index(invalid_index)
                , number_of_args(0)
            {}
        };

        llvm::SmallVector< vast_args, 8 > arg_info;

      public:
        clang_to_vast_arg_mapping(
            const acontext_t &actx, const function_info_t &fninfo, bool only_required_args = false
        )
            : inalloca_arg_no(invalid_index)
            , sret_arg_no(invalid_index)
            , total_vast_args(0)
            , arg_info(only_required_args ? fninfo.get_num_required_args() : fninfo.arg_size())
        {
            construct(actx, fninfo, only_required_args);
        }

        bool has_sret_arg() const { return sret_arg_no != invalid_index; }

        bool has_inalloca_arg() const { return inalloca_arg_no != invalid_index; }

        unsigned get_total_vast_args() const { return total_vast_args; }

        bool has_padding_arg(unsigned arg_idx) const {
            VAST_ASSERT(arg_idx < arg_info.size());
            return arg_info[arg_idx].padding_arg_index != invalid_index;
        }

        /// Returns index of first vast argument corresponding to arg_idx, and their
        /// quantity.
        std::pair< unsigned, unsigned > get_vast_args(unsigned arg_idx) const {
            VAST_ASSERT(arg_idx < arg_info.size());
            return std::make_pair(
                arg_info[arg_idx].first_arg_index, arg_info[arg_idx].number_of_args
            );
        }

      private:
        void construct(const acontext_t &Context, const function_info_t &fninfo, bool only_required_args);
    };

} // namespace vast::cg
