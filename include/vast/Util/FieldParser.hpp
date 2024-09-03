// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <mlir/IR/DialectImplementation.h>
VAST_UNRELAX_WARNINGS

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

