// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreTraits.hpp"
#include "vast/Interfaces/SymbolInterface.hpp"
#include "vast/Util/Common.hpp"

#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#define GET_OP_CLASSES
#include "vast/Dialect/Core/Core.h.inc"
