// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
VAST_UNRELAX_WARNINGS

namespace vast::tw {
    using conversion_path_t             = std::vector< std::string >;
    using conversion_path_fingerprint_t = std::string;

    using handle_id_t = std::size_t;

    struct handle_t
    {
        handle_id_t id;
        vast_module mod;
    };

} // namespace vast::tw
