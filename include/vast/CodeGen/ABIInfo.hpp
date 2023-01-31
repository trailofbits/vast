// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

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

        virtual ~abi_info_t() = default;

        virtual void compute_info(function_info_t &fninfo) const = 0;

        // Implement the Type::IsPromotableIntegerType for ABI specific needs. The
        // only difference is that this consideres bit-precise integer types as well.
        bool is_promotable_integer_type_for_abi(qual_type type) const;
    };

    struct aarch64_abi_info : abi_info_t {
        enum class abi_kind
        {
            aapccs = 0,
            darwin_pcs,
            win64
        };

        aarch64_abi_info(types_generator &types, abi_kind kind)
            : abi_info_t(types)
            , kind(kind)
        {}

        virtual ~aarch64_abi_info() = default;

      private:
        abi_kind get_abi_kind() const { return kind; }
        bool is_darwin_pcs() const { return kind == abi_kind::darwin_pcs; }

        abi_arg_info classify_return_type(qual_type rty, bool variadic) const;
        abi_arg_info classify_arg_type(
            qual_type rty, bool variadic, unsigned calling_convention
        ) const;

        void compute_info(function_info_t &fninfo) const override;

        abi_kind kind;
    };

    /// The AVX ABI leel for X86 targets.
    enum class x86_avx_abi_level { none, avx, avx512 };

    struct x86_64_abi_info : abi_info_t {
        x86_64_abi_info(types_generator &types, x86_avx_abi_level /* avx_level */)
            : abi_info_t(types)
            /* , avx_level(avx_level) */
        {}

        virtual ~x86_64_abi_info() = default;

        void compute_info(function_info_t &fninfo) const override;

        abi_arg_info classify_return_type(qual_type rty, bool variadic) const;
        abi_arg_info classify_arg_type(
            qual_type rty, bool variadic, unsigned calling_convention
        ) const;

      private:
        enum class abi_class
        {
            integer = 0,
            sse,
            sseup,
            x87,
            x87up,
            complex_x87,
            no_class,
            memory
        };
    };

    struct darwin_x86_64_abi_info : abi_info_t {
        darwin_x86_64_abi_info(types_generator &types, x86_avx_abi_level /* avx_level */)
            : abi_info_t(types)
            /* , avx_level(avx_level) */
        {}

        virtual ~darwin_x86_64_abi_info() = default;

        void compute_info(function_info_t &fninfo) const override;

        abi_arg_info classify_return_type(qual_type rty, bool variadic) const;
        abi_arg_info classify_arg_type(
            qual_type rty, bool variadic, unsigned calling_convention
        ) const;
    };

} // namespace vast::cg
