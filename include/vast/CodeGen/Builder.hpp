// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast::cg {

    using codegen_builder_base = mlir::OpBuilder;

    struct codegen_builder : codegen_builder_base {
        using base = codegen_builder_base;

        codegen_builder(mcontext_t &ctx) : base(&ctx) {}
    };

} // namespace vast::cg
