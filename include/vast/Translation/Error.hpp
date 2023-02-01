// Copyright (c) 2022-2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Except.hpp"

namespace vast::cg
{
    struct codegen_error : util::error {
        using util::error::error;
    };

    struct unimplemented : util::error {
        explicit unimplemented(std::string err, int exit = 1)
            : error("[codegen] not implemented: " + err, exit)
        {}
    };

} // namespace vast::cg
