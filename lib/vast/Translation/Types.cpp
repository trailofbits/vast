// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Translation/Types.hpp"
#include "vast/Dialect/HighLevel/HighLevel.hpp"

#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"

#include <iostream>

namespace vast::hl
{
    using builtin_type = clang::BuiltinType;

    constexpr integer_qualifier get_integer_qualifier(const builtin_type *ty)
    {
        return ty->isSignedInteger() ? integer_qualifier::Signed : integer_qualifier::Unsigned;
    }

    constexpr integer_kind get_integer_kind(const builtin_type *ty)
    {
        switch (ty->getKind()) {
            case builtin_type::SChar:
            case builtin_type::UChar:
                return integer_kind::Char;
            case builtin_type::Short:
            case builtin_type::UShort:
                return integer_kind::Short;
            case builtin_type::Int:
            case builtin_type::UInt:
                return integer_kind::Int;
            case builtin_type::Long:
            case builtin_type::ULong:
                return integer_kind::Long;
            case builtin_type::LongLong:
            case builtin_type::ULongLong:
                return integer_kind::LongLong;
            default:
                llvm_unreachable("unknown integer kind");
        }
    }

    constexpr bool is_void_type(const builtin_type *ty) { return ty->isVoidType(); }
    constexpr bool is_bool_type(const builtin_type *ty) { return ty->getKind() == builtin_type::Bool; }
    constexpr bool is_integer_type(const builtin_type *ty) { return ty->isIntegerType(); }

    mlir::Type TypeConverter::convert(clang::QualType ty)
    {
        return convert(ty.getTypePtr());
    }

    mlir::Type TypeConverter::convert(const clang::Type *ty)
    {
        ty = ty->getUnqualifiedDesugaredType();

        if (ty->isBuiltinType())
            return convert(clang::cast<builtin_type>(ty));

        llvm_unreachable("unknown clang type");
    }

    mlir::Type TypeConverter::convert(const builtin_type *ty)
    {
        // TODO(Heno) qualifiers
        if (is_void_type(ty)) {
            return VoidType::get(ctx);
        } else if (is_bool_type(ty)) {
            return BoolType::get(ctx);
        } else if (is_integer_type(ty)) {
            auto qual = get_integer_qualifier(ty);
            auto kind = get_integer_kind(ty);
            return IntegerType::get(ctx, qual, kind);
        }

        llvm_unreachable("unknown builtin type");
    }

    mlir::FunctionType TypeConverter::convert(const clang::FunctionType *ty)
    {
        llvm::SmallVector< mlir::Type, 2 > args;

        if (auto prototype = clang::dyn_cast< clang::FunctionProtoType >(ty)) {
            for (auto param : prototype->getParamTypes()) {
                args.push_back(convert(param));
            }
        }

        auto rty = convert(ty->getReturnType());
        return mlir::FunctionType::get(ctx, args, rty);
    }

} // namseapce vast::hl
