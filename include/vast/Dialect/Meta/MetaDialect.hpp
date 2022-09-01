// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
VAST_RELAX_WARNINGS


// Pull in the dialect definition.
#include "vast/Dialect/Meta/MetaDialect.h.inc"

namespace vast::meta
{
    using identifier_t = std::uint64_t;

    void add_identifier(mlir::Operation *op, identifier_t id);

    void remove_identifier(mlir::Operation *op);

    std::vector< mlir::Operation * > get_with_identifier(mlir::Operation *scope, identifier_t id);

    std::vector< mlir::Operation * > get_with_meta_location(mlir::Operation *scope, identifier_t id);

} // namespace vast::meta
