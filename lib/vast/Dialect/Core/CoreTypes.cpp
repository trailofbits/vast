// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Util/Common.hpp"

using StringRef = llvm::StringRef; // to fix missing namespace in generated file

namespace vast::core {
    void CoreDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/Core/CoreTypes.cpp.inc"
        >();
    }


    static ParseResult parseFunctionTypeArgs(
        mlir::AsmParser &parser, llvm::SmallVector< mlir::Type > &params, bool &isVarArg
    ) {
        isVarArg = false;
        // `(` `)`
        if (mlir::succeeded(parser.parseOptionalRParen())) {
            return mlir::success();
        }

        // `(` `...` `)`
        if (mlir::succeeded(parser.parseOptionalEllipsis())) {
            isVarArg = true;
            return parser.parseRParen();
        }

        // type (`,` type)* (`,` `...`)?
        mlir::Type type;
        if (parser.parseType(type)) {
            return mlir::failure();
        }
        params.push_back(type);
        while (mlir::succeeded(parser.parseOptionalComma())) {
            if (mlir::succeeded(parser.parseOptionalEllipsis())) {
                isVarArg = true;
                return parser.parseRParen();
            }
            if (parser.parseType(type)) {
                return mlir::failure();
            }
            params.push_back(type);
        }

        return parser.parseRParen();
    }

    static void printFunctionTypeArgs(
        mlir::AsmPrinter &printer, mlir::ArrayRef< mlir::Type > params, bool isVarArg
    ) {
        llvm::interleaveComma(params, printer, [&printer](mlir::Type type) {
            printer.printType(type);
        });
        if (isVarArg) {
            if (!params.empty()) {
                printer << ", ";
            }
            printer << "...";
        }
        printer << ')';
    }

    FunctionType FunctionType::clone(mlir::TypeRange inputs, mlir::TypeRange results) const {
        return get(llvm::to_vector(inputs), llvm::to_vector(results), isVarArg());
    }

    mlir::FunctionType lower(FunctionType fty) {
        // TODO what about varargs?
        return mlir::FunctionType::get(
            fty.getContext(), fty.getInputs(), fty.getResults()
        );
    }

} // namespace vast::core

VAST_RELAX_WARNINGS
#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/Core/CoreTypes.cpp.inc"
VAST_UNRELAX_WARNINGS
