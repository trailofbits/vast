// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "vast/Dialect/HighLevel/IR/HighLevelDialect.h.inc"

// #define GET_ATTRDEF_CLASSES
// #include "vast/Dialect/HighLevel/IR/HighLevelAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/IR/HighLevelTypes.h.inc"

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/IR/HighLevel.h.inc"