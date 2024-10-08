// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Dialect/Core/Func.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/FunctionImplementation.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Region.hpp"

namespace vast::ll
{
    mlir::SuccessorOperands Br::getSuccessorOperands( unsigned idx )
    {
        VAST_CHECK( idx == 0, "ll::Br can have only one successor!" );
        return mlir::SuccessorOperands( getOperandsMutable() );
    }

    // This is currently stolen from HighLevel/HighLevelOps.cpp.
    // Do we need a separate version?

    //===----------------------------------------------------------------------===//
    // FuncOp
    //===----------------------------------------------------------------------===//

    logical_result FuncOp::verify() {
        return core::verifyFuncOp(*this);
    }

    ParseResult parseFunctionSignatureAndBody(
        Parser &parser, Attribute &function_type,
        mlir::NamedAttrList &attr_dict, Region &body
    ) {
        return core::parseFunctionSignatureAndBody(parser, function_type, attr_dict, body);
    }

    void printFunctionSignatureAndBody(
        Printer &printer, FuncOp op, Attribute function_type,
        mlir::DictionaryAttr dict_attr, Region &body
    ) {
        return core::printFunctionSignatureAndBodyImpl(printer, op, function_type, dict_attr, body);
    }

} // namespace vast::ll

using vast::core::parseStorageClasses;
using vast::core::printStorageClasses;

#define GET_OP_CLASSES
#include "vast/Dialect/LowLevel/LowLevel.cpp.inc"
