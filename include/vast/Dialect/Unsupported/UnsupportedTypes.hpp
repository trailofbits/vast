// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Interfaces/DataLayoutInterfaces.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Interfaces/DefaultDataLayoutTypeInterface.hpp"

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/Unsupported/UnsupportedTypes.h.inc"
