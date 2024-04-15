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

#include "vast/Dialect/Core/CoreAttributes.hpp"
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

    using diagnostics_engine    = clang::DiagnosticsEngine;

    using virtual_file_system = llvm::vfs::FileSystem;

    struct action_options {
        const header_search_options &headers;
        const codegen_options &codegen;
        const target_options &target;
        const language_options &lang;
        const frontend_options &front;
        diagnostics_engine &diags;
        virtual_file_system &vfs;
    };

    using option_t = string_ref;

    constexpr option_t vast_option_prefix = "-vast-";

    struct vast_args
    {
        using option_list = std::vector< option_t >;

        argv_storage args;

        // detects the presence of an option in one of formats:
        // (1) -vast-"name"
        // (2) -vast-"name"="value"
        bool has_option(option_t opt) const;

        // from option of form -vast-"name"="value" returns the "value"
        std::optional< option_t > get_option(string_ref opt) const;

        // from option of form -vast-"name"="value1;value2;value3" returns list of values
        std::optional< option_list > get_options_list(string_ref opt) const;

        void push_back(arg_t arg);
    };

    std::pair< vast_args, argv_storage > filter_args(const argv_storage_base &args);

    namespace opt {
        constexpr option_t emit_llvm = "emit-llvm";
        constexpr option_t emit_obj  = "emit-obj";
        constexpr option_t emit_asm  = "emit-asm";
        constexpr option_t emit_mlir = "emit-mlir";

        constexpr option_t print_pipeline = "print-pipeline";
        constexpr option_t emit_crash_reproducer = "emit-crash-reproducer";

        constexpr option_t disable_multithreading = "disable-multithreading";
        constexpr option_t debug = "debug";

        constexpr option_t simplify = "simplify";
        constexpr option_t canonicalize = "canonicalize";

        llvm::Twine disable(string_ref pipeline_name);

        constexpr option_t show_locs = "show-locs";
        constexpr option_t locs_as_meta_ids = "locs-as-meta-ids";

        constexpr option_t disable_vast_verifier = "disable-vast-verifier";
        constexpr option_t vast_verify_diags = "verify-diags";
        constexpr option_t disable_emit_cxx_default = "disable-emit-cxx-default";

        bool emit_only_mlir(const vast_args &vargs);
        bool emit_only_llvm(const vast_args &vargs);
    } // namespace opt

    using source_language = core::SourceLanguage;

    source_language get_source_language(const language_options &opts);

} // namespace vast::cc
