// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/FunctionImplementation.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreDialect.hpp"

namespace vast::core {

    llvm::StringRef getLinkageAttrNameString();

    // This function is adapted from CIR:
    //
    // Verifies linkage types, similar to LLVM:
    // - functions don't have 'common' linkage
    // - external functions have 'external' or 'extern_weak' linkage
    logical_result verifyFuncOp(auto op) {
        using core::GlobalLinkageKind;

        auto linkage = op.getLinkage();
        constexpr auto common = GlobalLinkageKind::CommonLinkage;
        if (linkage == common) {
            return op.emitOpError() << "functions cannot have '"
                << stringifyGlobalLinkageKind(common)
                << "' linkage";
        }

        // isExternal(FunctionOpInterface) only checks for empty body...
        // We need to be able to handle functions with internal linkage without body.
        if (linkage != GlobalLinkageKind::InternalLinkage && op.isExternal()) {
            constexpr auto external = GlobalLinkageKind::ExternalLinkage;
            constexpr auto weak_external = GlobalLinkageKind::ExternalWeakLinkage;
            if (linkage != external && linkage != weak_external) {
                return op.emitOpError() << "external functions must have '"
                    << stringifyGlobalLinkageKind(external)
                    << "' or '"
                    << stringifyGlobalLinkageKind(weak_external)
                    << "' linkage";
            }
            return mlir::success();
        }
        return mlir::success();
    }

    ParseResult parseFunctionSignatureAndBody(
        Parser &parser, Attribute &funcion_type,
        mlir::NamedAttrList &attr_dict, Region &body
    );


    void printFunctionSignatureAndBody(
        Printer &printer, auto op,
        Attribute /* funcion_type */, mlir::DictionaryAttr, Region &body
    ) {
        printer << stringifyGlobalLinkageKind(op.getLinkage()) << ' ';

        auto fty = op.getFunctionType();
        mlir::function_interface_impl::printFunctionSignature(
            printer, op, fty.getInputs(), fty.isVarArg(), fty.getResults()
        );

        mlir::function_interface_impl::printFunctionAttributes(
            printer, op, { getLinkageAttrNameString(), op.getFunctionTypeAttrName() }
        );

        if (!body.empty()) {
            printer.getStream() << " ";
            printer.printRegion( body,
                /* printEntryBlockArgs */false,
                /* printBlockTerminators */true
            );
        }
    }
} // namespace vast::core
