// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_VASTOPS_HPP
#define VAST_VASTOPS_HPP

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "vast/Dialect/VastOps.h.inc"

#endif // VAST_VASTOPS_HPP