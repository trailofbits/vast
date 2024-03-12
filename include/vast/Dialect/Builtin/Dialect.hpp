// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Core/TypeTraits.hpp"
// Pull in the dialect definition.
#include "vast/Dialect/Builtin/BuiltinDialect.h.inc"
