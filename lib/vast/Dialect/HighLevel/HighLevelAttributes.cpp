// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

namespace vast::hl
{
    using Context = mlir::MLIRContext;

    Attribute parse_bool_attr(Context *ctx, DialectParser &parser)
    {
        if (parser.parseLess())
            return Attribute();

        bool value;
        if (succeeded(parser.parseOptionalKeyword("true")))
            value = true;
        else if (succeeded(parser.parseOptionalKeyword("false")))
            value = false;
        else
            return Attribute();

        if (parser.parseGreater())
            return Attribute();

        return HLBoolAttr::get(ctx, value);
    }

    void print_bool_attr(const HLBoolAttr &attr, DialectPrinter &printer)
    {
        auto value = attr.getValue() ? "true" : "false";
        printer << attr.getMnemonic() << "<" << value << ">";
    }

    template< typename IntegerAttr >
    Attribute parse_integer_attr(Context *ctx, DialectParser &parser)
    {
        if (parser.parseLess())
            return Attribute();

        llvm::APInt value;
        if (parser.parseInteger(value))
            return Attribute();

        if (parser.parseGreater())
            return Attribute();

        return IntegerAttr::get(ctx, value);
    }

    template< typename IntegerAttr >
    void print_integer_attr(const IntegerAttr &attr, DialectPrinter &printer)
    {
        printer << attr.getMnemonic() << "<" << attr.getValue() << ">";
    }

    template< typename FloatingAttr >
    Attribute parse_floating_attr(Context *ctx, DialectParser &parser)
    {
        if (parser.parseLess())
            return Attribute();

        double value;
        if (parser.parseFloat(value))
            return Attribute();

        if (parser.parseGreater())
            return Attribute();

        return FloatingAttr::get(ctx, llvm::APFloat(value));
    }

    template< typename FloatingAttr >
    void print_floating_attr(const FloatingAttr &attr, DialectPrinter &printer)
    {
        printer << attr.getMnemonic() << "<" << attr.getValue() << ">";
    }

    Attribute parse_str_attr(Context *ctx, DialectParser &parser)
    {
        return Attribute();
    }

    void print_str_attr(const StringAttr &attr, DialectPrinter &printer)
    {
        printer << attr.getMnemonic() << "<" << attr.getValue() << ">";
    }

} // namespace vast::hl

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"
namespace vast::hl
{
    using DialectParser = mlir::DialectAsmParser;
    using DialectPrinter = mlir::DialectAsmPrinter;

    void HighLevelDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"
        >();
    }

    Attribute HighLevelDialect::parseAttribute(DialectParser &parser, Type type) const
    {
        auto loc = parser.getCurrentLocation();

        llvm::StringRef mnemonic;
        if (parser.parseKeyword(&mnemonic))
            return Attribute();

        Attribute result;
        if (generatedAttributeParser(getContext(), parser, mnemonic, type, result).hasValue()) {
            return result;
        }

        parser.emitError(loc, "unexpected high-level attribute '" + mnemonic + "'");
        return Attribute();
    }

    void HighLevelDialect::printAttribute(Attribute attr, DialectPrinter &p) const
    {
        if (failed(generatedAttributePrinter(attr, p)))
            UNREACHABLE("Unexpected attribute");
    }

} // namespace vast::hl
