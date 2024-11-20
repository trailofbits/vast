// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Parser/Dialect.hpp"
#include "vast/Dialect/Parser/Types.hpp"
#include "vast/Util/Common.hpp"

#define GET_OP_CLASSES
#include "vast/Dialect/Parser/Parser.h.inc"
