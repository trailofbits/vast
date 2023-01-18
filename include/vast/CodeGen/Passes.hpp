// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"

namespace vast::cg {

    logical_result emit_high_level_pass(
        vast_module mod, mcontext_t *mctx, acontext_t *actx, bool enable_verifier
    );

} // namespace vast::cg
