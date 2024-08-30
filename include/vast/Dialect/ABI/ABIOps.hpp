// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>

#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
VAST_RELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/Func.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"

#define GET_OP_CLASSES
#include "vast/Dialect/ABI/ABI.h.inc"
