// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

namespace vast::cc {

    enum class target_dialect { high_level, std, llvm, cir };

    target_dialect parse_target_dialect(string_ref from);

    std::string to_string(target_dialect target);

} // namespace vast::cc
