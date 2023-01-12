// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Except.hpp"

namespace vast::cc
{
    template< typename T >
    using llvm_cnt_ptr = llvm::IntrusiveRefCntPtr< T >;

    struct compiler_error : util::error {
        using util::error::error;
    };

    using arg_t  = const char *;
    using argv_t = llvm::ArrayRef< arg_t >;

    using argv_storage = llvm::SmallVector< arg_t, 256 >;

} // namespace vast::cc
