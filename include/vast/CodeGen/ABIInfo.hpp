// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <clang/AST/Type.h>
VAST_UNRELAX_WARNINGS

#include <vast/CodeGen/FunctionInfo.hpp>

namespace vast::cg
{
    struct types_generator;

    /// abi_info_t - Target specific hooks for defining how a type should be passed or
    /// returned from functions.
    class abi_info_t {
        abi_info_t() = delete;

      public:
        types_generator &types;

        abi_info_t(types_generator &types)
            : types{ types }
        {}

        virtual ~abi_info_t();

        virtual void compute_info(function_info_t &fninfo) const = 0;

        // Implement the Type::IsPromotableIntegerType for ABI specific needs. The
        // only difference is that this consideres bit-precise integer types as well.
        bool is_promotable_integer_type_for_abi(clang::QualType type) const;
    };

} // namespace vast::cg
