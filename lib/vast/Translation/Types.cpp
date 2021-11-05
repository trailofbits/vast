// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Translation/Types.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"

#include <iostream>

namespace vast::hl
{
    using BuiltinType = clang::BuiltinType;

    constexpr Signedness get_signedness_qualifier(const BuiltinType *ty)
    {
        return ty->isSignedInteger() ? Signedness::Signed : Signedness::Unsigned;
    }

    constexpr IntegerKind get_integer_kind(const BuiltinType *ty)
    {
        switch (ty->getKind()) {
            case BuiltinType::Char_U:
            case BuiltinType::UChar:
            case BuiltinType::Char_S:
            case BuiltinType::SChar:
                return IntegerKind::Char;
            case BuiltinType::Short:
            case BuiltinType::UShort:
                return IntegerKind::Short;
            case BuiltinType::Int:
            case BuiltinType::UInt:
                return IntegerKind::Int;
            case BuiltinType::Long:
            case BuiltinType::ULong:
                return IntegerKind::Long;
            case BuiltinType::LongLong:
            case BuiltinType::ULongLong:
                return IntegerKind::LongLong;
            default:
                llvm_unreachable("unknown integer kind");
        }
    }

    constexpr bool is_void_type(const BuiltinType *ty)    { return ty->isVoidType(); }
    constexpr bool is_bool_type(const BuiltinType *ty)    { return ty->getKind() == BuiltinType::Bool; }
    constexpr bool is_integer_type(const BuiltinType *ty) { return ty->isIntegerType(); }

    std::vector< Qualifier > qualifiers_list(const BuiltinType *ty, clang::Qualifiers quals)
    {
        std::vector< Qualifier > qualifiers;
        if (ty->isUnsignedInteger() && !is_bool_type(ty))
            qualifiers.push_back(Signedness::Unsigned);
        if (quals.hasConst())
            qualifiers.push_back(Const());
        if (quals.hasVolatile())
            qualifiers.push_back(Volatile());
        return qualifiers;
    }

    mlir::Type TypeConverter::convert(clang::QualType ty)
    {
        return convert(ty.getTypePtr(), ty.getQualifiers());
    }

    mlir::Type TypeConverter::convert(const clang::Type *ty, clang::Qualifiers quals)
    {
        ty = ty->getUnqualifiedDesugaredType();

        if (ty->isBuiltinType())
            return convert(clang::cast<BuiltinType>(ty), quals);

        llvm_unreachable("unknown clang type");
    }

    mlir::Type TypeConverter::convert(const BuiltinType *ty, clang::Qualifiers quals)
    {
        if (is_void_type(ty)) {
            return VoidType::get(ctx);
        }

        if (is_bool_type(ty)) {
            return BoolType::get(ctx, qualifiers_list(ty, quals));
        }

        if (is_integer_type(ty)) {
            auto kind = get_integer_kind(ty);
            return IntegerType::get(ctx, kind, qualifiers_list(ty, quals));
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
