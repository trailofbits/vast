// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
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

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/Core/CoreAttributes.cpp.inc"

namespace vast::core
{
    bool is_core_typed_attr(mlir::Attribute attr) {
        return util::is_one_of< typed_attrs >(attr);
    }

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

    void CoreDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/Core/CoreAttributes.cpp.inc"
        >();
    }

} // namespace vast::core
