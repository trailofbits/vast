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

    void HighLevelDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"
        >();
    }

} // namespace vast::hl
