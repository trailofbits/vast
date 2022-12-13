// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
VAST_RELAX_WARNINGS

#include "vast/Interfaces/SymbolInterface.hpp"
#include "vast/Dialect/HighLevel/HighLevelLinkage.hpp"

#define GET_OP_CLASSES
#include "vast/Dialect/ABI/ABI.h.inc"

