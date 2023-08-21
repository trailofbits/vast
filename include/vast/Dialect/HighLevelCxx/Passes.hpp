// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevelCxx/HighLevelCxxDialect.hpp"

namespace vast::hl::cxx
{
    #define GEN_PASS_REGISTRATION
    #include "vast/Dialect/HighLevelCxx/Passes.h.inc"

} // namespace vast::hl::cxx