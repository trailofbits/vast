// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include <sstream>

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

namespace vast::hl
{
    namespace detail
    {
        template< typename TypeList, std::size_t... Idx >
        bool is_one_of(mlir::Type type, std::index_sequence<Idx...>)
        {
            return (type.isa< std::tuple_element_t< Idx, TypeList > >() || ...);
        }

        template< typename TypeList >
        bool is_one_of(mlir::Type type)
        {
            constexpr auto length = std::tuple_size_v< TypeList >;
            return is_one_of< TypeList >( type, std::make_index_sequence< length >{} );
        }
    }

    mlir::FunctionType getFunctionType(PointerType functionPointer)
    {
        return functionPointer.getElementType().cast< mlir::FunctionType >();
    }

    mlir::FunctionType getFunctionType(mlir::Type functionPointer)
    {
        return getFunctionType(functionPointer.cast< PointerType >());
    }

    void HighLevelDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"
        >();
    }

    using IntegerTypes = std::tuple<
        CharType, ShortType, IntType, LongType, LongLongType, Int128Type
    >;

    using FloatingTypes = std::tuple<
        FloatType, DoubleType, LongDoubleType
    >;

    bool isBoolType(mlir::Type type)
    {
        return type.isa< BoolType >();
    }

    bool isIntegerType(mlir::Type type)
    {
        return detail::is_one_of< IntegerTypes >(type);
    }

    bool isFloatingType(mlir::Type type)
    {
        return detail::is_one_of< FloatingTypes >(type);
    }

    template< typename CVType >
    Type parse_cv_type(Context *ctx, DialectParser &parser)
    {
        if (failed(parser.parseOptionalLess())) {
            return CVType::get(ctx);
        }

        bool c = succeeded(parser.parseOptionalKeyword("const"));
        bool v = succeeded(parser.parseOptionalKeyword("volatile"));

        auto loc = parser.getCurrentLocation();

        if (failed(parser.parseGreater())) {
            parser.emitError(loc, "expected end of qualifier list");
            return Type();
        }

        return CVType::get(ctx, c, v);
    }

    template< typename CVType >
    void print_cv_type(const CVType &type, DialectPrinter &printer)
    {
        printer << type.getMnemonic();

        if ( !(type.getIsVolatile() || type.getIsConst()) ) {
            return;
        }

        bool first = true;
        auto print = [&] (auto qual) {
            printer << (!first ? " " : "") << qual;
            first = false;
        };

        printer << "<";
        if (type.isConst())    { print("const"); }
        if (type.isVolatile()) { print("volatile"); }
        printer << ">";
    }

    template< typename IntegerType >
    Type parse_integer_type(Context *ctx, DialectParser &parser)
    {
        if (failed(parser.parseOptionalLess())) {
            return IntegerType::get(ctx);
        }

        bool u = succeeded(parser.parseOptionalKeyword("unsigned"));
        bool c = succeeded(parser.parseOptionalKeyword("const"));
        bool v = succeeded(parser.parseOptionalKeyword("volatile"));

        auto loc = parser.getCurrentLocation();

        if (failed(parser.parseGreater())) {
            parser.emitError(loc, "expected end of qualifier list");
            return Type();
        }

        return IntegerType::get(ctx, u, c, v);
    }

    template< typename IntegerType >
    void print_integer_type(const IntegerType &type, DialectPrinter &printer)
    {
        printer << type.getMnemonic();

        if ( !(type.getIsVolatile() || type.getIsConst() || type.getIsUnsigned()) ) {
            return;
        }

        bool first = true;
        auto print = [&] (auto qual) {
            printer << (!first ? " " : "") << qual;
            first = false;
        };

        printer << "<";
        if (type.isUnsigned()) { print("unsigned"); }
        if (type.isConst())    { print("const"); }
        if (type.isVolatile()) { print("volatile"); }
        printer << ">";
    }

    Type parse_pointer_type(Context *ctx, DialectParser &parser)
    {
        auto loc = parser.getCurrentLocation();
        if (failed(parser.parseLess())) {
            parser.emitError(loc, "expected <");
            return Type();
        }

        Type element;
        if (failed(parser.parseType(element))) {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "expected element type");
            return Type();
        }

        bool c = false, v = false;
        if (succeeded(parser.parseOptionalComma())) {
            c = succeeded(parser.parseOptionalKeyword("const"));
            v = succeeded(parser.parseOptionalKeyword("volatile"));
        }


        if (failed(parser.parseGreater())) {
            parser.emitError(loc, "expected end of qualifier list");
            return Type();
        }

        return PointerType::get(ctx, element, c, v);
    }

    void print_pointer_type(const PointerType &type, DialectPrinter &printer)
    {
        printer << type.getMnemonic() << "<";
        printer.printType(type.getElementType());

        if ( type.getIsVolatile() || type.getIsConst() ) {
            printer << ",";
            if (type.isConst())    { printer << " const"; }
            if (type.isVolatile()) { printer << " volatile"; }
        }

        printer << ">";
    }

} // namespace vast::hl


#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"

namespace vast::hl
{
    Type HighLevelDialect::parseType(DialectParser &parser) const
    {
        auto loc = parser.getCurrentLocation();
        llvm::StringRef mnemonic;
        if (parser.parseKeyword(&mnemonic))
            return Type();

        Type result;
        if (generatedTypeParser(getContext(), parser, mnemonic, result).hasValue()) {
            return result;
        }

        parser.emitError(loc, "unknown high-level type");
        return Type();
    }

    void HighLevelDialect::printType(Type type, DialectPrinter &os) const
    {
        if (failed(generatedTypePrinter(type, os)))
            UNREACHABLE("unexpected high-level type kind");
    }

} // namespace vast::hl
