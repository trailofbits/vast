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

#include <variant>

#include "vast/Util/Parser.hpp"

namespace vast::hl
{
    using type = mlir::Type;

    /* void type */
    struct void_mnemonic {};

    constexpr std::string_view to_string(void_mnemonic) { return "void"; }

    /* bool type */
    struct bool_mnemonic {};

    constexpr std::string_view to_string(bool_mnemonic) { return "bool"; }

    /* integer types */
    enum class integer_kind { Char, Short, Int, Long, LongLong };

    constexpr std::string_view to_string(integer_kind kind)
    {
        switch (kind) {
            case integer_kind::Char:     return "char";
            case integer_kind::Short:    return "short";
            case integer_kind::Int:      return "int";
            case integer_kind::Long:     return "long";
            case integer_kind::LongLong: return "longlong";
        }
    }

    /* floating point types */
    enum class floating_kind { Float, Double, LongDouble };

    constexpr std::string_view to_string(floating_kind kind)
    {
        switch (kind) {
            case floating_kind::Float:      return "float";
            case floating_kind::Double:     return "double";
            case floating_kind::LongDouble: return "longdouble";
        }
    }

    /* qualifiers */
    struct volatile_qualifier {};

    constexpr std::string_view to_string(volatile_qualifier) { return "volatile"; }

    struct const_qualifier {};

    constexpr std::string_view to_string(const_qualifier) { return "const"; }

    enum class signedness_qualifier { Signed, Unsigned };

    constexpr std::string_view to_string(signedness_qualifier qual)
    {
        switch (qual) {
            case signedness_qualifier::Signed:   return "signed";
            case signedness_qualifier::Unsigned: return "unsigned";
        }
    }

    /* variants of possible qualifiers */
    using qualifier = std::variant< volatile_qualifier, const_qualifier, signedness_qualifier >;

    constexpr std::string_view to_string(const qualifier &qual);

    /* variants of possible type mnemonics */
    using mnemonic = std::variant< integer_kind, floating_kind >;

    constexpr std::string_view to_string(const mnemonic &mnem);

    /* possible type name tokens */
    using token = std::variant< qualifier, mnemonic >;

    constexpr std::string_view to_string(const token &tok);

    namespace detail
    {
        static constexpr auto to_string = [] (const auto &v) { return vast::hl::to_string(v); };
    }

    constexpr std::string_view to_string(const qualifier &qual)
    {
        return std::visit(detail::to_string, qual);
    }

    constexpr std::string_view to_string(const mnemonic &mnem)
    {
        return std::visit(detail::to_string, mnem);
    }

    constexpr std::string_view to_string(const token &tok)
    {
        return std::visit(detail::to_string, tok);
    }

    template< typename stream >
    auto operator<<(stream &os, const token &tok) -> decltype(os << "")
    {
        return os << to_string(tok);
    }

    /* helper parsers */

    template< typename enum_type >
    constexpr parser< enum_type > auto enum_parser(enum_type value)
    {
        return as_enum(value, string_parser( to_string(value) ));
    }

    template< typename trivial >
    constexpr parser< trivial > auto trivial_parser()
    {
        return as_enum( trivial(), string_parser( to_string(trivial()) ));
    }

    /* type mnemonic parsers */
    constexpr parser< integer_kind > auto integer_kind_parser()
    {
        return enum_parser( integer_kind::Char  ) |
               enum_parser( integer_kind::Short ) |
               enum_parser( integer_kind::Int   ) |
               enum_parser( integer_kind::Long  ) |
               enum_parser( integer_kind::LongLong );
    }

    constexpr parser< floating_kind > auto float_kind_parser()
    {
        return enum_parser( floating_kind::Float  ) |
               enum_parser( floating_kind::Double ) |
               enum_parser( floating_kind::LongDouble );
    }

    constexpr parser< mnemonic > auto mnemonic_parser()
    {
        auto integer  = construct< mnemonic >( integer_kind_parser() );
        auto floating = construct< mnemonic >( float_kind_parser() );
        return integer | floating;
    }

    /* qualifier parsers */
    constexpr parser< signedness_qualifier > auto signedness_qualifier_parser()
    {
        return enum_parser( signedness_qualifier::Signed ) |
               enum_parser( signedness_qualifier::Unsigned );
    }

    constexpr parser< qualifier > auto qualifier_parser()
    {
        auto con = construct< qualifier >( trivial_parser< const_qualifier >() );
        auto vol = construct< qualifier >( trivial_parser< volatile_qualifier >() );
        auto sig = construct< qualifier >( signedness_qualifier_parser() );

        return con | vol | sig;
    }

    /* top level type name token parser */
    constexpr parser< token > auto type_parser()
    {
        auto mnem = construct< token >( mnemonic_parser() );
        auto qual = construct< token >( qualifier_parser() );
        return mnem | qual;
    }

} // namespace vast::hl
