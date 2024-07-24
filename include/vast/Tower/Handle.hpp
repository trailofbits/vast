// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

namespace vast::tw {
    using conversion_path_t             = std::vector< std::string >;
    using conversion_path_fingerprint_t = std::string;

    using handle_id_t = std::size_t;

    struct handle_t
    {
        handle_id_t id;
        core::module mod;
    };

} // namespace vast::tw
