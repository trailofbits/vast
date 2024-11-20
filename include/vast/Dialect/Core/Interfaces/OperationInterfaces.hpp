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

/// Include the generated interface declarations.
#include "vast/Dialect/Core/Interfaces/OperationInterfaces.h.inc"
