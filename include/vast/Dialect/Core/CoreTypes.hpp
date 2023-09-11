// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/Core/CoreDialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/Core/CoreTypes.h.inc"

namespace vast::core {

    mlir::FunctionType lower(core::FunctionType fty);

} // namespace vast::core
