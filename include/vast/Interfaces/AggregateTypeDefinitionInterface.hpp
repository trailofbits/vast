// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include <gap/coro/generator.hpp>

namespace vast {

    struct field_info_t
    {
        std::string name;
        mlir_type type;
    };

} // namespace vast


/// Include the generated interface declarations.
#include "vast/Interfaces/AggregateTypeDefinitionInterface.h.inc"
