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

    IntegerType IntegerType::get(Context *ctx, IntegerKind kind) { return Base::get(ctx, kind); }

    IntegerType IntegerType::get(Context *ctx, IntegerKind kind, QualifiersList qualifiers)
    {
        return Base::get(ctx, kind, qualifiers);
    }

    FloatingType FloatingType::get(Context *ctx, FloatingKind kind) { return Base::get(ctx, kind); }

    FloatingType FloatingType::get(Context *ctx, FloatingKind kind, QualifiersList qualifiers)
    {
        return Base::get(ctx, kind, qualifiers);
    }

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

    std::string to_string(HighLevelType type)
    {
        auto print = [&] (auto type) { return to_string(type); };

        return llvm::TypeSwitch< HighLevelType, std::string >(type)
            .Case< VoidType, BoolType, IntegerType, FloatingType >(print)
            .Default([&](auto) { return llvm_unreachable("unknown high-level type"), "invalid"; });
    }

    void HighLevelDialect::registerTypes() {
        addTypes< VoidType, BoolType, IntegerType, FloatingType >();

        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"
        >();
    }
} // namespace vast::hl
