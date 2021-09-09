// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.h.inc"

namespace vast::hl
{
    // fixes missing namespace in the tablegen code
    using Value = mlir::Value;

} // namespace vast::hl

// #define GET_ATTRDEF_CLASSES
// #include "vast/Dialect/HighLevel/HighLevelAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.h.inc"

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.h.inc"
