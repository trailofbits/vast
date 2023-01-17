// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Options.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/STLExtras.h>
VAST_UNRELAX_WARNINGS


namespace vast::cc {

    namespace detail {

        // drops "-vast" prefix from option string
        string_ref name_and_value_view(string_ref opt) {
            return opt.drop_front(vast_option_prefix.size());
        }

        std::optional< string_ref > get_option_impl(argv_t args, string_ref name) {
            auto is_opt_with_name = [] (auto name) {
                return [name] (auto arg) {
                    return name_and_value_view(arg).startswith(name);
                };
            };

            if (auto it = llvm::find_if(args, is_opt_with_name(name)); it != args.end()) {
                return string_ref(*it).drop_front(vast_option_prefix.size());
            }

            return std::nullopt;
        }
    } // detail

    bool vast_args::has_option(string_ref name) const {
        return detail::get_option_impl(args, name).has_value();
    }

    std::optional< string_ref > vast_args::get_option(string_ref name) const {
        if (auto opt = detail::get_option_impl(args, name)) {
            if (auto [lhs, rhs] = opt->split('='); !rhs.empty()) {
                return rhs;
            }
        }

        return std::nullopt;
    }

    void vast_args::push_back(arg_t arg) {
        args.push_back(arg);
    }

    std::pair< vast_args, argv_storage > filter_args(const argv_storage &args) {
        vast_args vargs;
        argv_storage rest;

        for (auto arg : args) {
            if (std::string_view(arg).starts_with(vast_option_prefix)) {
                vargs.push_back(arg);
            } else {
                rest.push_back(arg);
            }
        }

        return { vargs, rest };
    }

} // namespace vast::cc
