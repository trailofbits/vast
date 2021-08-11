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
    enum class type_kind {
        vast_integer,
        vast_floating,
        vast_void
    };

    enum class integer_qualifier {
        vast_signed,
        vast_unsigned
    };

    enum class integer_kind {
        vast_bool,
        vast_char,
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

    template< typename ConcreteType, typename BaseType, typename StorageType, template <typename T> class ...Traits >
    using type_base = type::TypeBase< ConcreteType, BaseType, StorageType, Traits... >;

    namespace detail
    {
        using default_type_storage = mlir::TypeStorage;

        struct integer_type_storage;
    } // namespace detail

    struct void_type : type_base< void_type, type, detail::default_type_storage >
    {
        using Base::Base;

        static void_type get(context *ctx);

        static bool kindof(unsigned kind) noexcept
        {
            return type_kind(kind) == type_kind::vast_void;
        }
    };

    // struct integer_type : type_base< integer_type, type, detail::integer_type_storage >
    // {
    //     using Base::Base;

    //     static integer_type get(context *ctx, integer_qualifier qual, integer_kind kind);

    //     static bool kindof(unsigned kind) noexcept
    //     {
    //         return type_kind(kind) == type_kind::vast_integer;
    //     }
    // };

} // namespace vast::hl