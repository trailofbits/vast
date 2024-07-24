// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

namespace vast::cg
{
    void emit_data_layout(mcontext_t &ctx, core::module mod, const dl::DataLayoutBlueprint &dl);

} // namespace vast::cg
