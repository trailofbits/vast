// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Util/FieldParser.hpp"
#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

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

    mlir::CallInterfaceCallable get_callable_for_callee(operation op) {
        return op->getAttrOfType< mlir::FlatSymbolRefAttr >("callee");
    }

} // namespace vast::core
