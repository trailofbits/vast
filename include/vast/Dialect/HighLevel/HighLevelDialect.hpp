// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Core/CoreTraits.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

// Pull in the dialect definition.
#include "vast/Dialect/HighLevel/HighLevelDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "vast/Dialect/HighLevel/HighLevelEnums.h.inc"


