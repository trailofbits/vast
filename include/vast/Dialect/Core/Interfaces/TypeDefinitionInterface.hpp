// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <llvm/ADT/StringRef.h>
VAST_RELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include <gap/coro/generator.hpp>

namespace vast::core {

    struct field_info_t
    {
        mlir::FlatSymbolRefAttr name;
        mlir_type type;
    };

} // namespace vast::core

/// Include the generated interface declarations.
#include "vast/Dialect/Core/Interfaces/TypeDefinitionInterface.h.inc"

namespace vast::core {

    using type_definition_interface = TypeDefinitionInterface;
    using aggregate_interface = AggregateTypeDefinitionInterface;

} // namespace vast::core
