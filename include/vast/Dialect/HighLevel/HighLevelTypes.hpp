// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>

#include <clang/AST/Type.h>

#include <llvm/ADT/Hashing.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Parser.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.h.inc"

#include <set>

namespace vast::hl
{
    using Type = mlir::Type;
    using Context = mlir::MLIRContext;

    /* void type */
    struct VoidMnemonic {};

    inline std::string to_string(VoidMnemonic) { return "void"; }

    /* bool type */
    struct BoolMnemonic {};

    inline std::string to_string(BoolMnemonic) { return "bool"; }

    /* integer types */
    enum class IntegerKind { Char, Short, Int, Long, LongLong };

    inline std::string to_string(IntegerKind kind)
    {
        switch (kind) {
            case IntegerKind::Char:     return "char";
            case IntegerKind::Short:    return "short";
            case IntegerKind::Int:      return "int";
            case IntegerKind::Long:     return "long";
            case IntegerKind::LongLong: return "longlong";
        }
    }

    /* floating point types */
    enum class FloatingKind { Float, Double, LongDouble };

    inline std::string to_string(FloatingKind kind)
    {
        switch (kind) {
            case FloatingKind::Float:      return "float";
            case FloatingKind::Double:     return "double";
            case FloatingKind::LongDouble: return "longdouble";
        }
    }

    /* pointer type */
    struct PointerMnemonic {};

    inline std::string to_string(PointerMnemonic) { return "ptr"; }

    /* qualifiers */

    /* volatile qualifier */
    struct Volatile {};

    inline std::string to_string(Volatile) { return "volatile"; }
    constexpr llvm::hash_code hash_value(Volatile) { return llvm::hash_code(); }

    constexpr bool operator==(const Volatile& lhs, const Volatile& rhs) { return true; }

    /* const qualifier */
    struct Const {};

    inline std::string to_string(Const) { return "const"; }
    constexpr llvm::hash_code hash_value(Const) { return llvm::hash_code(); }

    constexpr bool operator==(const Const& lhs, const Const& rhs) { return true; }

    /* signedness qualifier */
    enum class Signedness { Signed, Unsigned };

    inline std::string to_string(Signedness qual)
    {
        switch (qual) {
            case Signedness::Signed:   return "signed";
            case Signedness::Unsigned: return "unsigned";
        }
    }

    constexpr llvm::hash_code hash_value(const Signedness &qual)
    {
        return llvm::hash_value(qual);
    }

    /* variants of possible qualifiers */
    using Qualifier = std::variant< Volatile, Const, Signedness >;

    std::string to_string(const Qualifier &qual);

    constexpr llvm::hash_code hash_value(const Qualifier &qual);

    /* variants of possible type mnemonics */
    using Mnemonic = std::variant< VoidMnemonic, BoolMnemonic, PointerMnemonic, IntegerKind, FloatingKind >;

    std::string to_string(const Mnemonic &mnem);

    /* possible type name tokens */
    using TypeToken = std::variant< Qualifier, Mnemonic >;

    std::string to_string(const TypeToken &token);

    namespace detail
    {
        static constexpr auto to_string  = [] (const auto &v) { return vast::hl::to_string(v); };
        static constexpr auto hash_value = [] (const auto &v) { return vast::hl::hash_value(v); };
    }

    inline std::string to_string(const Qualifier &qual)
    {
        return std::visit(detail::to_string, qual);
    }

    constexpr llvm::hash_code hash_value(const Qualifier &qual)
    {
        return std::visit(detail::hash_value, qual);
    }

    inline std::string to_string(const Mnemonic &mnem)
    {
        return std::visit(detail::to_string, mnem);
    }

    inline std::string to_string(const TypeToken &tok)
    {
        return std::visit(detail::to_string, tok);
    }

    template< typename stream >
    auto operator<<(stream &os, const TypeToken &tok) -> decltype(os << "")
    {
        return os << to_string(tok);
    }

    /* helper parsers */

    template< typename enum_type >
    constexpr parser< enum_type > auto enum_parser(enum_type value)
    {
        return [value] (parse_input_t in) {
            auto str = to_string(value);
            return as_trivial(value, string_parser(str))(in);
        };
    }

    template< typename trivial >
    constexpr parser< trivial > auto trivial_parser()
    {
        return [value = trivial()] (parse_input_t in) {
            auto str = to_string(value);
            return as_trivial(value, string_parser(str))(in);
        };
    }

    /* type mnemonic parsers */
    constexpr parser< IntegerKind > auto integer_kind_parser()
    {
        return enum_parser( IntegerKind::Char  ) |
               enum_parser( IntegerKind::Short ) |
               enum_parser( IntegerKind::Int   ) |
               enum_parser( IntegerKind::LongLong ) |
               enum_parser( IntegerKind::Long  );
    }

    constexpr parser< FloatingKind > auto float_kind_parser()
    {
        return enum_parser( FloatingKind::Float  ) |
               enum_parser( FloatingKind::Double ) |
               enum_parser( FloatingKind::LongDouble );
    }

    constexpr parser< Mnemonic > auto mnemonic_parser()
    {
        auto _void    = construct< Mnemonic >( trivial_parser< VoidMnemonic >() );
        auto boolean  = construct< Mnemonic >( trivial_parser< BoolMnemonic >() );
        auto pointer  = construct< Mnemonic >( trivial_parser< PointerMnemonic >() );
        auto integer  = construct< Mnemonic >( integer_kind_parser() );
        auto floating = construct< Mnemonic >( float_kind_parser() );
        return _void | boolean | pointer | floating | integer;
    }

    /* qualifier parsers */
    constexpr parser< Signedness > auto signedness_parser()
    {
        return enum_parser( Signedness::Signed ) |
               enum_parser( Signedness::Unsigned );
    }

    constexpr parser< Qualifier > auto qualifier_parser()
    {
        auto con = construct< Qualifier >( trivial_parser< Const >() );
        auto vol = construct< Qualifier >( trivial_parser< Volatile >() );
        auto sig = construct< Qualifier >( signedness_parser() );

        return con | vol | sig;
    }

    /* top level type name token parser */
    constexpr parser< TypeToken > auto type_parser()
    {
        auto mnem = construct< TypeToken >( mnemonic_parser() );
        auto qual = construct< TypeToken >( qualifier_parser() );
        return mnem | qual;
    }

    /* MLIR Type definitions */

    using TypeStorage = mlir::TypeStorage;

    using TypeStorageAllocator = mlir::TypeStorageAllocator;

    struct HighLevelType : Type
    {
        /// Return true if this is a 'ground' type, aka a non-aggregate type.
        bool isGround();

        /// Support method to enable LLVM-style type casting.
        static bool classof(Type type)
        {
            return llvm::isa< HighLevelDialect >( type.getDialect() );
        }

        protected:
            using Type::Type;
    };

    using QualifiersList = llvm::ArrayRef< Qualifier >;

    template< typename ...Qualifiers >
    struct QualifiersStorage : TypeStorage
    {
        using KeyTy = QualifiersList;

        QualifiersStorage(QualifiersList qualifiers)
            : qualifiers(qualifiers)
        {}

        bool operator==(const KeyTy &key) const { return key == qualifiers; }

        static QualifiersStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key)
        {
            QualifiersList qualifiers = allocator.copyInto(key);
            return new (allocator.allocate<QualifiersStorage>()) QualifiersStorage(qualifiers);
        }

        static llvm::hash_code hashKey(const KeyTy &key) { return llvm::hash_value(key); }

        QualifiersList qualifiers;
    };

    template< typename Value, typename ...Qualifiers >
    struct ValueWithQualifiersStorage : TypeStorage
    {
        using KeyTy = std::tuple< Value, QualifiersList >;

        explicit ValueWithQualifiersStorage(Value value)
            : value(value)
        {}

        ValueWithQualifiersStorage(Value value, QualifiersList qualifiers)
            : value(value), qualifiers(qualifiers)
        {}

        bool operator==(const KeyTy &key) const{ return key == KeyTy(value, qualifiers); }

        static ValueWithQualifiersStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key)
        {
            const auto &[value, quals] = key;
            QualifiersList qualifiers = allocator.copyInto(quals);
            return new (allocator.allocate<ValueWithQualifiersStorage>()) ValueWithQualifiersStorage(value, qualifiers);
        }

        static llvm::hash_code hashKey(const KeyTy &key) { return llvm::hash_value(key); }

        Value value;
        QualifiersList qualifiers;
    };


    template< typename Derived, typename Storage = TypeStorage >
    struct WithStorage : HighLevelType::TypeBase< Derived, HighLevelType, Storage >
    {
        using Base = HighLevelType::TypeBase< Derived, HighLevelType, Storage >;
        using Base::Base;
    };

    template< typename Kind, typename Next >
    struct WithKind : Next
    {
        using Next::Next;
        Kind kind() const { return this->getImpl()->value; }

        Mnemonic mnemonic() const { return kind(); }
    };

    template< typename Next >
    struct WithQualifiers : Next
    {
        using Next::Next;
        QualifiersList qualifiers() const { return this->getImpl()->qualifiers; }
    };

    /* Void Type */
    struct VoidType : WithStorage< VoidType >
    {
        using Base = WithStorage< VoidType >;
        using Base::Base;

        Mnemonic mnemonic() const { return VoidMnemonic{}; }

        static VoidType get(Context *ctx);
    };

    std::string to_string(VoidType type);

    /* Bool Type */
    using BoolStorage = QualifiersStorage< Const, Volatile >;

    struct BoolType : WithQualifiers< WithStorage< BoolType, BoolStorage > >
    {
        using Base = WithQualifiers< WithStorage< BoolType, BoolStorage > >;
        using Base::Base;

        using Base::qualifiers;

        Mnemonic mnemonic() const { return BoolMnemonic{}; }

        static BoolType get(Context *ctx);

        static BoolType get(Context *ctx, QualifiersList qualifiers);
    };

    std::string to_string(BoolType type);

    /* Integer Types */
    using IntegerStorage = ValueWithQualifiersStorage< IntegerKind, Const, Volatile, Signedness >;

    struct IntegerType : WithKind< IntegerKind, WithQualifiers< WithStorage< IntegerType, IntegerStorage > > >
    {
        using Base = WithKind< IntegerKind, WithQualifiers< WithStorage< IntegerType, IntegerStorage > > >;
        using Base::Base;

        using Base::kind;
        using Base::mnemonic;
        using Base::qualifiers;

        static IntegerType get(Context *ctx, IntegerKind kind);
        static IntegerType get(Context *ctx, IntegerKind kind, QualifiersList qualifiers);
    };

    std::string to_string(IntegerType type);

    /* Floating Types */
    using FloatingStorage = ValueWithQualifiersStorage< FloatingKind, Const, Volatile, Signedness >;

    struct FloatingType : WithKind< FloatingKind, WithQualifiers< WithStorage< FloatingType, FloatingStorage > > >
    {
        using Base = WithKind< FloatingKind, WithQualifiers< WithStorage< FloatingType, FloatingStorage > > >;
        using Base::Base;

        using Base::kind;
        using Base::mnemonic;
        using Base::qualifiers;

        static FloatingType get(Context *ctx, FloatingKind kind);
        static FloatingType get(Context *ctx, FloatingKind kind, QualifiersList qualifiers);
    };

    std::string to_string(FloatingType type);

    /* Pointer Type */
    using PointerStorage = ValueWithQualifiersStorage< HighLevelType, Const, Volatile >;

    struct PointerType : WithQualifiers< WithStorage< PointerType, PointerStorage > >
    {
        using Base = WithQualifiers< WithStorage< PointerType, PointerStorage > >;
        using Base::Base;

        using Base::qualifiers;

        Mnemonic mnemonic() const { return PointerMnemonic{}; }

        HighLevelType getElementType() const { return this->getImpl()->value; }

        static PointerType get(Context *ctx, HighLevelType elementType);
        static PointerType get(Context *ctx, HighLevelType elementType, QualifiersList qualifiers);
    };

    std::string to_string(PointerType type);

    std::string to_string(HighLevelType type);

} // namespace vast::hl
