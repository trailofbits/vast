// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/Interfaces/FunctionInterface.hpp"

#define GET_OP_FWD_DEFINES
#include "vast/Dialect/HighLevel/HighLevel.h.inc"
/// Include the generated interface declarations.
#include "vast/Dialect/Core/Interfaces/DeclStorageInterface.h.inc"
