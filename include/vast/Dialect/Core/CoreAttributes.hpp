// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/APSInt.h>
#include <llvm/Support/Locale.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Interfaces/CallInterfaces.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreTraits.hpp"

#include "vast/Interfaces/TypeQualifiersInterfaces.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolRefInterface.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/TypeList.hpp"

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/Core/CoreAttributes.h.inc"

namespace vast::core {

    mlir::CallInterfaceCallable get_callable_for_callee(operation op);

    ParseResult parseStorageClasses(
        Parser &parser, Attribute &storage_class, Attribute &thread_storage_class
    );

    void printStorageClasses(
        Printer &printer, mlir::Operation *op, core::StorageClassAttr storage_class, core::TSClassAttr thread_storage_class
    );

} // namespace vast::core
