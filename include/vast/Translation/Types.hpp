// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

namespace vast::cg
{
    using mlir_type = mlir::Type;

    using qual_type = clang::QualType;
    using qual_types_span = llvm::ArrayRef< qual_type >;

} // namespace vast::cg
