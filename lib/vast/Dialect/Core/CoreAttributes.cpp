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

    ParseResult parseStorageClasses(
        Parser &parser, Attribute &storage_class, Attribute &thread_storage_class
    ) {
        std::string keyword;
        auto parse_result = parser.parseOptionalKeywordOrString(&keyword);

        if (mlir::failed(parse_result)) {
            storage_class = core::StorageClassAttr::get(parser.getContext(), core::StorageClass::sc_none);
            thread_storage_class = core::TSClassAttr::get(parser.getContext(), core::TSClass::tsc_none);
            return mlir::success();
        } else if (auto attr = symbolizeEnum< core::StorageClass >(keyword)) {
            storage_class = core::StorageClassAttr::get(parser.getContext(), attr.value());
        } else if (auto attr = symbolizeEnum< core::TSClass >(keyword)) {
            storage_class = core::StorageClassAttr::get(parser.getContext(), core::StorageClass::sc_none);
            thread_storage_class = core::TSClassAttr::get(parser.getContext(), attr.value());
            return mlir::success();
        } else {
            return mlir::failure();
        }

        parse_result = parser.parseOptionalKeywordOrString(&keyword);
        if (mlir::failed(parse_result)) {
            thread_storage_class = core::TSClassAttr::get(parser.getContext(), core::TSClass::tsc_none);
        } else if (auto attr = symbolizeEnum< core::TSClass >(keyword)) {
            thread_storage_class = core::TSClassAttr::get(parser.getContext(), attr.value());
        } else {
            return mlir::failure();
        }

        return mlir::success();
    }

    void printStorageClasses(
        Printer &printer, mlir::Operation *op, core::StorageClassAttr storage_class, core::TSClassAttr thread_storage_class
    ) {
        if (storage_class.getValue() != core::StorageClass::sc_none) {
            printer << ' ' << storage_class.getValue();
        }

        if (thread_storage_class.getValue() != core::TSClass::tsc_none) {
            printer << ' ' << thread_storage_class.getValue();
        }
    }

} // namespace vast::core

MLIR_DEFINE_EXPLICIT_TYPE_ID(vast::core::VarSymbolRefAttr);
MLIR_DEFINE_EXPLICIT_TYPE_ID(vast::core::TypeSymbolRefAttr);
MLIR_DEFINE_EXPLICIT_TYPE_ID(vast::core::FuncSymbolRefAttr);
MLIR_DEFINE_EXPLICIT_TYPE_ID(vast::core::LabelSymbolRefAttr);
MLIR_DEFINE_EXPLICIT_TYPE_ID(vast::core::EnumConstantSymbolRefAttr);
MLIR_DEFINE_EXPLICIT_TYPE_ID(vast::core::MemberVarSymbolRefAttr);
MLIR_DEFINE_EXPLICIT_TYPE_ID(vast::core::ElaboratedTypeSymbolRefAttr);
