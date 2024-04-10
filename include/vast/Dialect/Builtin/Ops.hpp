// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Builtin/Dialect.hpp"
#include "vast/Util/Common.hpp"

#define GET_OP_CLASSES
#include "vast/Dialect/Builtin/Builtin.h.inc"
