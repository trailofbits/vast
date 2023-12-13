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

    void CoreDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/Core/CoreAttributes.cpp.inc"
        >();
    }

} // namespace vast::core
