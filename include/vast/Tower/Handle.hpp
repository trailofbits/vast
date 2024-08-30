// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

namespace vast::tw {
    // These two should most likely be unified as one type.
    using conversion_passes_t           = std::vector< pass_ptr >;

    using conversion_path_t             = std::vector< std::string >;
    using conversion_path_fingerprint_t = std::string;

    using handle_id_t = std::size_t;

    struct handle_t
    {
        handle_id_t id;
        mlir_module mod;
    };

} // namespace vast::tw
