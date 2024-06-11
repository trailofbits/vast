// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/CXXInheritance.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/Common.hpp"

namespace vast::cg
{
    struct meta_generator {
        virtual ~meta_generator() = default;
        virtual loc_t location(const clang_decl *) const = 0;
        virtual loc_t location(const clang_stmt *) const = 0;
        virtual loc_t location(const clang_expr *) const = 0;
    };

} // namespace vast::cg
