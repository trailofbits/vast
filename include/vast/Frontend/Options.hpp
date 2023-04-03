// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Frontend/FrontendOptions.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast::cc
{
    using arg_t  = const char *;
    using argv_t = llvm::ArrayRef< arg_t >;

    using argv_storage = llvm::SmallVector< arg_t, 256 >;
    using argv_storage_base = llvm::SmallVectorImpl< arg_t >;

    using header_search_options = clang::HeaderSearchOptions;
    using codegen_options       = clang::CodeGenOptions;
    using target_options        = clang::TargetOptions;
    using language_options      = clang::LangOptions;
    using frontend_options      = clang::FrontendOptions;

    constexpr string_ref vast_option_prefix = "-vast-";

    struct vast_args
    {
        using option_list = std::vector< string_ref >;
        using maybe_option_list = std::optional< option_list >;

        argv_storage args;

        // detects the presence of an option in one of formats:
        // (1) -vast-"name"
        // (2) -vast-"name"="value"
        bool has_option(string_ref opt) const;

        // from option of form -vast-"name"="value" returns the "value"
        std::optional< string_ref > get_option(string_ref opt) const;

        // from option of form -vast-"name"="value1;value2;value3" returns list of values
        maybe_option_list get_options_list(string_ref opt) const;

        void push_back(arg_t arg);
    };

    std::pair< vast_args, argv_storage > filter_args(const argv_storage_base &args);

} // namespace vast::cc
