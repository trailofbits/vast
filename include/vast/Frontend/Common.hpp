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

#include "vast/Util/Except.hpp"

namespace vast::cc
{
    template< typename T >
    using llvm_cnt_ptr = llvm::IntrusiveRefCntPtr< T >;

    struct compiler_error : util::error {
        using util::error::error;
    };

} // namespace vast::cc
