// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/Common.hpp"

namespace vast::cg {

    using symbol_name = string_ref;

    struct symbol_generator
    {
        virtual ~symbol_generator() = default;

        virtual std::optional< symbol_name > symbol(clang_global decl) = 0;
        virtual std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) = 0;
    };

} // namespace vast::cg
