// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Translation/Types.hpp"
#include "vast/Dialect/HighLevel/HighLevel.hpp"

#include "clang/AST/Type.h"

namespace vast::hl
{
    using builtin_type = clang::BuiltinType;

    constexpr integer_qualifier get_integer_qualifier(const builtin_type *ty)
    {
        return ty->isSignedInteger() ? integer_qualifier::vast_signed : integer_qualifier::vast_unsigned;
    }

    constexpr integer_kind get_integer_kind(const builtin_type *ty)
    {
        switch (ty->getKind()) {
            case builtin_type::SChar:
            case builtin_type::UChar:
                return integer_kind::vast_char;
            case builtin_type::Short:
            case builtin_type::UShort:
                return integer_kind::vast_short;
            case builtin_type::Int:
            case builtin_type::UInt:
                return integer_kind::vast_int;
            case builtin_type::Long:
            case builtin_type::ULong:
                return integer_kind::vast_long;
            case builtin_type::LongLong:
            case builtin_type::ULongLong:
                return integer_kind::vast_long_long;
            default:
                llvm_unreachable("unknown integer kind");
        }
    }

    constexpr bool is_void_type(const builtin_type *ty) { return ty->isVoidType(); }
    constexpr bool is_bool_type(const builtin_type *ty) { return ty->getKind() == builtin_type::Bool; }
    constexpr bool is_integer_type(const builtin_type *ty) { return ty->isIntegerType(); }

    mlir::Type TypeConverter::convert(const builtin_type *ty)
    {
        // TODO(Heno) qualifiers
        if (is_bool_type(ty)) {
            return VoidType::get(&ctx);
        } else if (is_bool_type(ty)) {
            return BoolType::get(&ctx);
        } else if (is_integer_type(ty)) {
            auto qual = get_integer_qualifier(ty);
            auto kind = get_integer_kind(ty);
            return IntegerType::get(&ctx, qual, kind);
        }

        llvm_unreachable("unknown builtin type");
    }

} // namseapce vast::hl