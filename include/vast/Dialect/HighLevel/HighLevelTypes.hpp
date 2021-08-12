// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"

namespace vast::hl
{
    enum class integer_qualifier {
        vast_signed,
        vast_unsigned
    };

    enum class integer_kind {
        vast_short,
        vast_int,
        vast_long,
        vast_long_long
    };

    enum class floating_kind {
        vast_float,
        vast_double,
        vast_long_double
    };

    using context = mlir::MLIRContext;
    using type = mlir::Type;

} // namespace vast::hl