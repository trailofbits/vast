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
    enum class signedness_qualifier {
        Signed,
        Unsigned
    };

    enum class integer_kind {
        Char,
        Short,
        Int,
        Long,
        LongLong
    };

    enum class floating_kind {
        Float,
        Double,
        LongDouble
    };

    using context = mlir::MLIRContext;
    using type = mlir::Type;

    std::string to_string(integer_kind kind) noexcept;

} // namespace vast::hl
