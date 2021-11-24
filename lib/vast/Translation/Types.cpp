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
            case BuiltinType::Int128:
            case BuiltinType::UInt128:
                return IntegerKind::Int128;
            default:
                UNREACHABLE("unknown integer kind");
        }
    }

    std::vector< Qualifier > qualifiers_list(const clang::Type *ty, clang::Qualifiers quals)
    {
        std::vector< Qualifier > qualifiers;
        if (ty->isUnsignedIntegerType() && !ty->isBooleanType())
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

    std::string TypeConverter::format_type(const clang::Type *type) const
    {
        std::string name;
        llvm::raw_string_ostream os(name);
        type->dump(os, actx);
        return name;
    }

    mlir::Type TypeConverter::convert(const clang::Type *ty, clang::Qualifiers quals)
    {
        ty = ty->getUnqualifiedDesugaredType();

        if (ty->isBuiltinType())
            return convert(clang::cast< BuiltinType >(ty), quals);

        if (ty->isPointerType())
            return convert(clang::cast< clang::PointerType >(ty), quals);

        if (ty->isRecordType())
            return convert(clang::cast< clang::RecordType >(ty), quals);

        if (ty->isArrayType())
            return convert(clang::cast< clang::ArrayType >(ty), quals);

        if (ty->isFunctionType())
            return convert(clang::cast< clang::FunctionType >(ty));

        UNREACHABLE( "unknown clang type: {0}", format_type(ty) );
    }

    mlir::Type TypeConverter::convert(const BuiltinType *ty, clang::Qualifiers quals)
    {
        if (ty->isVoidType()) {
            return VoidType::get(mctx);
        }

        if (ty->isBooleanType()) {
            return BoolType::get(mctx, qualifiers_list(ty, quals));
        }

        if (ty->isIntegerType()) {
            auto kind = get_integer_kind(ty);
            return IntegerType::get(mctx, kind, qualifiers_list(ty, quals));
        }

        UNREACHABLE( "unknown builtin type: {0}", format_type(ty) );
    }

    mlir::Type TypeConverter::convert(const clang::PointerType *ty, clang::Qualifiers quals)
    {
        auto elementType = convert(ty->getPointeeType());
        return PointerType::get(mctx, elementType, qualifiers_list(ty, quals));
    }

    mlir::Type TypeConverter::convert(const clang::RecordType *ty, clang::Qualifiers quals)
    {
        return RecordType::get(mctx);
    }

    mlir::Type TypeConverter::convert(const clang::ArrayType *ty, clang::Qualifiers quals)
    {
        return ArrayType::get(mctx);
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
        return mlir::FunctionType::get(mctx, args, rty);
    }

} // namseapce vast::hl
