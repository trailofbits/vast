// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

namespace vast::cg
{
    using mlir_type = mlir::Type;

    using qual_type = clang::QualType;
    using qual_types_span = llvm::ArrayRef< qual_type >;

    using ext_param_info = clang::FunctionProtoType::ExtParameterInfo;

    using ext_info = clang::FunctionType::ExtInfo;
    using ext_parameter_info_span = llvm::ArrayRef< ext_param_info >;

} // namespace vast::cg
