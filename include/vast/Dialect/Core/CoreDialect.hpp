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
#include "vast/Dialect/Core/CoreDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "vast/Dialect/Core/CoreEnums.h.inc"
