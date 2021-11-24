// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include <sstream>

#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"

#include <llvm/ADT/TypeSwitch.h>

namespace vast::hl
{
    namespace detail
    {
        using Storage = mlir::TypeStorage;

        std::string to_string_qualifiers(QualifiersList qualifiers)
        {
            std::stringstream ss;

            auto print = [&] (const auto &qual) { ss << qual; };
            auto space = [&] { ss << " "; };
            llvm::interleave(qualifiers, print, space);

            return ss.str();
        }

        std::string to_string_with_qualifiers(const auto &type)
        {
            auto qualifiers = type.qualifiers();

            auto result = to_string(type.mnemonic());
            if (qualifiers.empty())
                return result;

            return result + "<" + to_string_qualifiers(qualifiers) + ">";
        }

    } // namespace detail

    bool HighLevelType::isGround()
    {
        return llvm::TypeSwitch< HighLevelType, bool >( *this )
            .Case< VoidType, BoolType, IntegerType, FloatingType >( [] (Type) { return true; } )
            .Default( [] (Type) { llvm_unreachable("unknown high-level type"); return false; } );
    }

    VoidType VoidType::get(Context *ctx) { return Base::get(ctx); }

    BoolType BoolType::get(Context *ctx) { return Base::get(ctx); }

    BoolType BoolType::get(Context *ctx, QualifiersList qualifiers)
    {
        return Base::get(ctx, qualifiers);
    }

    IntegerType IntegerType::get(Context *ctx, IntegerKind kind)
    {
        return Base::get(ctx, kind, QualifiersList());
    }

    IntegerType IntegerType::get(Context *ctx, IntegerKind kind, QualifiersList qualifiers)
    {
        return Base::get(ctx, kind, qualifiers);
    }

    FloatingType FloatingType::get(Context *ctx, FloatingKind kind)
    {
        return Base::get(ctx, kind, QualifiersList());
    }

    FloatingType FloatingType::get(Context *ctx, FloatingKind kind, QualifiersList qualifiers)
    {
        return Base::get(ctx, kind, qualifiers);
    }

    PointerType PointerType::get(Context *ctx, mlir::Type elementType)
    {
        return Base::get(ctx, elementType, QualifiersList());
    }

    PointerType PointerType::get(Context *ctx, mlir::Type elementType, QualifiersList qualifiers)
    {
        return Base::get(ctx, elementType, qualifiers);
    }

    RecordType RecordType::get(Context *ctx) { return Base::get(ctx); }

    ArrayType ArrayType::get(Context *ctx) { return Base::get(ctx); }

    std::string to_string(VoidType type)
    {
        return to_string(type.mnemonic());
    }

    std::string to_string(BoolType type)
    {
        return detail::to_string_with_qualifiers(type);
    }

    std::string to_string(IntegerType type)
    {
        return detail::to_string_with_qualifiers(type);
    }

    std::string to_string(FloatingType type)
    {
        return detail::to_string_with_qualifiers(type);
    }

    std::string to_string(mlir::FunctionType type)
    {
        std::string name;
        llvm::raw_string_ostream os(name);
        type.print(os);
        return name;
    }

    std::string to_string(mlir::Type type)
    {
        if (auto hlty = type.dyn_cast< HighLevelType >())
            return to_string(hlty);
        if (auto fty = type.dyn_cast< mlir::FunctionType >())
            return to_string(fty);
        llvm_unreachable( "unsupported type" );
    }

    std::string to_string(PointerType type)
    {
        auto repr = to_string(type.mnemonic()) + "<" + to_string(type.getElementType());
        auto qualifiers = type.qualifiers();
        if (!qualifiers.empty()) {
            repr += ", " + detail::to_string_qualifiers(qualifiers);
        }
        return repr + ">";
    }

    mlir::FunctionType getFunctionType(PointerType functionPointer)
    {
        return functionPointer.getElementType().cast< mlir::FunctionType >();
    }

    mlir::FunctionType getFunctionType(mlir::Type functionPointer)
    {
        return getFunctionType(functionPointer.cast< PointerType >());
    }

    std::string to_string(RecordType type)
    {
        return to_string(type.mnemonic());
    }

    std::string to_string(ArrayType type)
    {
        return to_string(type.mnemonic());
    }

    std::string to_string(HighLevelType type)
    {
        auto print = [&] (auto type) { return to_string(type); };

        return llvm::TypeSwitch< HighLevelType, std::string >(type)
            .Case< VoidType, BoolType, IntegerType, FloatingType, PointerType, RecordType, ArrayType >(print)
            .Default([&](auto) { return llvm_unreachable("unknown high-level type"), "invalid"; });
    }

    void HighLevelDialect::registerTypes() {
        addTypes< VoidType, BoolType, IntegerType, FloatingType, PointerType, RecordType, ArrayType >();

        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"
        >();
    }
} // namespace vast::hl
