// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>

#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/IR/FunctionInterfaces.h>
VAST_RELAX_WARNINGS

#include "vast/Interfaces/SymbolInterface.hpp"
#include "vast/Interfaces/ElementTypeInterface.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Util/Common.hpp"

#define GET_OP_CLASSES
#include "vast/Dialect/LowLevel/LowLevel.h.inc"
