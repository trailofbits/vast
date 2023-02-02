// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Translation/Types.hpp"
namespace vast::cg
{
    using ext_param_info = clang::FunctionProtoType::ExtParameterInfo;

    using ext_info = clang::FunctionType::ExtInfo;
    using ext_parameter_info_span = llvm::ArrayRef< ext_param_info >;

} // namespace vast::cg


