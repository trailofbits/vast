// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Type.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"
#include "vast/Util/TypeList.hpp"
#include "vast/Util/Types.hpp"

namespace vast::us {

    mlir::Type strip_unsupported(mlir::Type);
    mlir::Type strip_unsupported(mlir::Value);

} // namespace vast::us

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/Unsupported/UnsupportedTypes.h.inc"
