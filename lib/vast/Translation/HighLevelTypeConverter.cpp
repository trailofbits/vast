// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Translation/HighLevelTypeConverter.hpp"

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include <cassert>
#include <iostream>

namespace vast::hl
{
    using BuiltinType = clang::BuiltinType;

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

    constexpr FloatingKind get_floating_kind(const BuiltinType *ty)
    {
        switch (ty->getKind()) {
            case BuiltinType::Half:
            case BuiltinType::Float16:
                return FloatingKind::Half;
            case BuiltinType::BFloat16:
                return FloatingKind::BFloat16;
            case BuiltinType::Float:
                return FloatingKind::Float;
            case BuiltinType::Double:
                return FloatingKind::Double;
            case BuiltinType::LongDouble:
                return FloatingKind::LongDouble;
            case BuiltinType::Float128:
                return FloatingKind::Float128;
            default:
                UNREACHABLE("unknown floating kind");
        }
    }

    bool DataLayoutBlueprint::try_emplace(mlir::Type mty, const clang::Type *aty,
                                          const clang::ASTContext &actx)
    {
        // NOTE(lukas): clang changes size of `bool` to `1` when emitting llvm.
        if (aty->isBooleanType())
        {
            return std::get< 1 >(entries.try_emplace(mty, dl::DLEntry{ mty, 1 }));
        }

        // For other types this should be good-enough for now
        auto info = actx.getTypeInfo(aty);
        auto bw = static_cast< uint32_t >(info.Width);
        const auto &[_, flag] = entries.try_emplace(mty, dl::DLEntry{ mty, bw });
        return flag;
    }

    mlir::Type HighLevelTypeConverter::convert(clang::QualType ty) {
        return convert(ty.getTypePtr(), ty.getQualifiers());
    }

    mlir::Type HighLevelTypeConverter::convert(const clang::Type *ty, clang::Qualifiers quals) {
        return dl_aware_convert(ty, quals);
    }

    mlir::Type HighLevelTypeConverter::dl_aware_convert(
        const clang::Type *ty, clang::Qualifiers quals) {
        auto out = do_convert(ty, quals);
        dl.try_emplace(out, ty, ctx.getASTContext());
        return out;
    }

    mlir::Type HighLevelTypeConverter::convert(const clang::RecordType *ty, bool definition) {
        auto decl = ty->getDecl();
        CHECK(decl->getIdentifier(), "anonymous records not supported yet");
        auto name = decl->getName();

        auto declared = ctx.lookup_typedecl(name);
        auto mctx     = &ctx.getMLIRContext();

        if (definition || !declared) {
            llvm::SmallVector< FieldInfo, 2 > fields;
            for (const auto &field : decl->fields()) {
                auto field_name = mlir::StringAttr::get(mctx, field->getName());
                auto field_type = convert(field->getType());
                fields.push_back(FieldInfo{ field_name, field_type });
            }

            return RecordType::get(mctx, fields);
        }

        return AliasType::get(mctx, mlir::SymbolRefAttr::get(mctx, name));
    }

    std::string HighLevelTypeConverter::format_type(const clang::Type *type) const {
        std::string name;
        llvm::raw_string_ostream os(name);
        type->dump(os, ctx.getASTContext());
        return name;
    }

    mlir::Type HighLevelTypeConverter::do_convert(const clang::Type *ty, clang::Qualifiers quals) {
        ty = ty->getUnqualifiedDesugaredType();

        if (ty->isBuiltinType())
            return do_convert(clang::cast< BuiltinType >(ty), quals);

        if (ty->isPointerType())
            return do_convert(clang::cast< clang::PointerType >(ty), quals);

        if (ty->isRecordType())
            return do_convert(clang::cast< clang::RecordType >(ty), quals);

        if (ty->isConstantArrayType())
            return do_convert(clang::cast< clang::ConstantArrayType >(ty), quals);

        if (ty->isFunctionType())
            return convert(clang::cast< clang::FunctionType >(ty));

        UNREACHABLE( "unknown clang type: {0}", format_type(ty) );
    }

    mlir::Type HighLevelTypeConverter::do_convert(const BuiltinType *ty, clang::Qualifiers quals) {
        auto v = quals.hasVolatile();
        auto c = quals.hasConst();

        auto &mctx = ctx.getMLIRContext();

        if (ty->isVoidType()) {
            return VoidType::get(&mctx);
        }

        if (ty->isBooleanType()) {
            return BoolType::get(&mctx, c, v);
        }

        if (ty->isIntegerType()) {
            auto u = ty->isUnsignedIntegerType();

            switch (get_integer_kind(ty)) {
                case IntegerKind::Char:     return CharType::get(&mctx, u, c, v);
                case IntegerKind::Short:    return ShortType::get(&mctx, u, c, v);
                case IntegerKind::Int:      return IntType::get(&mctx, u, c, v);
                case IntegerKind::Long:     return LongType::get(&mctx, u, c, v);
                case IntegerKind::LongLong: return LongLongType::get(&mctx, u, c, v);
                case IntegerKind::Int128:   return Int128Type::get(&mctx, u, c, v);
            }
        }

        if (ty->isFloatingType()) {
            switch (get_floating_kind(ty)) {
                case FloatingKind::Half:       return HalfType::get(&mctx, c, v);
                case FloatingKind::BFloat16:   return BFloat16Type::get(&mctx, c, v);
                case FloatingKind::Float:      return FloatType::get(&mctx, c, v);
                case FloatingKind::Double:     return DoubleType::get(&mctx, c, v);
                case FloatingKind::LongDouble: return LongDoubleType::get(&mctx, c, v);
                case FloatingKind::Float128:   return Float128Type::get(&mctx, c, v);
            }
        }

        UNREACHABLE( "unknown builtin type: {0}", format_type(ty) );
    }

    mlir::Type HighLevelTypeConverter::do_convert(
        const clang::PointerType *ty, clang::Qualifiers quals) {
        auto pointee = convert(ty->getPointeeType());
        return PointerType::get(
            &ctx.getMLIRContext(), pointee, quals.hasConst(), quals.hasVolatile());
    }

    mlir::Type HighLevelTypeConverter::do_convert(
        const clang::RecordType *ty, clang::Qualifiers quals) {
        return RecordType::get(&ctx.getMLIRContext());
    }

    mlir::Type HighLevelTypeConverter::do_convert(
        const clang::ConstantArrayType *ty, clang::Qualifiers quals) {
        assert(clang::isa< clang::ConstantArrayType >(ty));
        auto element_type = convert(ty->getElementType());
        auto size = ty->getSize();
        return ConstantArrayType::get(
            &ctx.getMLIRContext(), element_type, size, quals.hasConst(), quals.hasVolatile());
    }

    mlir::FunctionType HighLevelTypeConverter::convert(const clang::FunctionType *ty) {
        llvm::SmallVector< mlir::Type, 2 > args;

        if (auto prototype = clang::dyn_cast< clang::FunctionProtoType >(ty)) {
            for (auto param : prototype->getParamTypes()) {
                args.push_back(convert(param));
            }
        }

        auto rty = convert(ty->getReturnType());
        return mlir::FunctionType::get(&ctx.getMLIRContext(), args, rty);
    }

} // namseapce vast::hl
