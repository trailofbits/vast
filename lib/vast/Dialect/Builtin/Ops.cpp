// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/Support/ErrorHandling.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Builtin/Dialect.hpp"
#include "vast/Dialect/Builtin/Ops.hpp"
#include "vast/Dialect/Builtin/Types.hpp"

using namespace vast::hlbi;

#define GET_OP_CLASSES
#include "vast/Dialect/Builtin/Builtin.cpp.inc"
