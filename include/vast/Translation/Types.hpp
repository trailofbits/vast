// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

namespace vast::cg
{
    using mlir_type = mlir::Type;

    using qual_type = clang::QualType;
    using can_qual_type = clang::CanQualType;
    using can_qual_types_span = llvm::ArrayRef< can_qual_type >;

} // namespace vast::cg
