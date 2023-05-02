// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

namespace mlir {

    template<>
    struct FieldParser<bool> {
        static FailureOr<bool> parse(AsmParser &parser) {
            if (succeeded(parser.parseOptionalKeyword("true"))) {
                return true;
            }
            if (succeeded(parser.parseOptionalKeyword("false"))) {
                return false;
            }
            return failure();
        }
    };

    template<>
    struct FieldParser<llvm::APInt> {
        static FailureOr<llvm::APInt> parse(AsmParser &parser) {
            llvm::APInt value;
            if (parser.parseInteger(value))
                return failure();
            return value;
        }
    };

    template<>
    struct FieldParser<llvm::APSInt> {
        static FailureOr<llvm::APSInt> parse(AsmParser &parser) {
            llvm::APInt value;
            if (parser.parseInteger(value))
                return failure();
            return llvm::APSInt(value, false);
        }
    };


    template<>
    struct FieldParser<llvm::APFloat> {
        static FailureOr<llvm::APFloat> parse(AsmParser &parser) {
            // TODO fix float parser
            double value;
            if (parser.parseFloat(value))
                return failure();
            return llvm::APFloat(value);
        }
    };

} // namespace mlir

namespace vast::hl
{
    using Context = mlir::MLIRContext;

} // namespace vast::hl

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"

namespace vast::hl
{
    using DialectParser = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

    // This function is a carbon-copy of TableGen generated function for StringAttr
    // With the exception of re-escaping the string
    Attribute StringLiteralAttr::parse(DialectParser &parser, mlir::Type attrType)
    {
        mlir::Builder odsBuilder(parser.getContext());
        llvm::SMLoc odsLoc = parser.getCurrentLocation();
        (void) odsLoc;
        // Parse literal '<'
        if (parser.parseLess()) return {};

        // Parse variable 'value'
        auto parsed_str = mlir::FieldParser<std::string>::parse(parser);
        if (::mlir::failed(parsed_str)) {
          parser.emitError(parser.getCurrentLocation(),
          "failed to parse StringAttr parameter 'value' which is to be a `::llvm::StringRef`");
          return {};
        }

        // because AsmParser can't output raw string...
        std::string res = escapeString(llvm::StringRef(*parsed_str));

        // Parse literal '>'
        if (parser.parseGreater()) return {};
        assert(::mlir::succeeded(parsed_str));
        return StringLiteralAttr::get(parser.getContext(),
            ::llvm::StringRef(res),
            ::mlir::Type(attrType));
    }

    // This function is a carbon-copy of TableGen generated function for StringAttr
    void StringLiteralAttr::print(DialectPrinter &printer) const {
        mlir::Builder odsBuilder(getContext());
        printer << "<";
        printer << '"' << getValue() << '"';;
        printer << ">";
    }

    void HighLevelDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"
        >();
    }

} // namespace vast::hl
