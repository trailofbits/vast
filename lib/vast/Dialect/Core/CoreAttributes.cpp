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
    using DialectParser = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

    Attribute StringLiteralAttr::parse(DialectParser &parser, mlir_type attr_type)
    {
        mlir_builder builder(parser.getContext());
        // Parse literal '<'
        if (parser.parseLess()) return {};

        // Parse variable 'value'
        auto parsed_str = mlir::FieldParser<std::string>::parse(parser);
        if (::mlir::failed(parsed_str)) {
          parser.emitError(parser.getCurrentLocation(),
          "failed to parse StringAttr parameter 'value' which is to be a `::llvm::StringRef`");
          return {};
        }

        // Parse literal '>'
        if (parser.parseGreater()) return {};

        // Automatically generated parser for `AnyAttr` might pass default
        // constructed `mlir::Type` instead of `mlir::NoneType`…
        // If we simply pass the dummy type inside the attribute becomes
        // unprintable
        if (!attr_type)
            attr_type = builder.getType< mlir::NoneType >();

        return StringLiteralAttr::get(parser.getContext(),
            ::llvm::StringRef(*parsed_str),
            ::mlir::Type(attr_type));
    }

    void StringLiteralAttr::print(DialectPrinter &printer) const {
        auto escaped = escapeString(getValue());
        printer << "<";
        printer << '"' << escaped << '"';
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
