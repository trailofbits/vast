// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
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

    // This function is adapted from CIR:
    //
    // Verifies linkage types, similar to LLVM:
    // - functions don't have 'common' linkage
    // - external functions have 'external' or 'extern_weak' linkage
    logical_result FuncOp::verify() {
        using core::GlobalLinkageKind;

        auto linkage = getLinkage();
        constexpr auto common = GlobalLinkageKind::CommonLinkage;
        if (linkage == common) {
            return emitOpError() << "functions cannot have '"
                << stringifyGlobalLinkageKind(common)
                << "' linkage";
        }

        // isExternal(FunctionOpInterface) only checks for empty bodyonly checks for empty body...
        // We need to be able to handle functions with internal linkage without body.
        if (linkage != GlobalLinkageKind::InternalLinkage && isExternal()) {
            constexpr auto external = GlobalLinkageKind::ExternalLinkage;
            constexpr auto weak_external = GlobalLinkageKind::ExternalWeakLinkage;
            if (linkage != external && linkage != weak_external) {
                return emitOpError() << "external functions must have '"
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
        Parser &parser, Attribute &funcion_type, mlir::NamedAttrList &attr_dict, Region &body
    ) {
        llvm::SmallVector< Parser::Argument, 8 > arguments;
        llvm::SmallVector< mlir::DictionaryAttr, 1 > result_attrs;
        llvm::SmallVector< Type, 8 > arg_types;
        llvm::SmallVector< Type, 4 > result_types;

        auto &builder = parser.getBuilder();

        bool is_variadic = false;
        if (mlir::failed(mlir::function_interface_impl::parseFunctionSignature(
            parser, /*allowVariadic=*/false, arguments, is_variadic, result_types, result_attrs
        ))) {
            return mlir::failure();
        }


        for (auto &arg : arguments) {
            arg_types.push_back(arg.type);
        }

        // create parsed function type
        funcion_type = mlir::TypeAttr::get(
            builder.getFunctionType(arg_types, result_types)
        );

        // If additional attributes are present, parse them.
        if (parser.parseOptionalAttrDictWithKeyword(attr_dict)) {
            return mlir::failure();
        }

        // TODO: Add the attributes to the function arguments.
        // VAST_ASSERT(result_attrs.size() == result_types.size());
        // return mlir::function_interface_impl::addArgAndResultAttrs(
        //     builder, state, arguments, result_attrs
        // );

        auto loc = parser.getCurrentLocation();
        auto parse_result = parser.parseOptionalRegion(
            body, arguments, /* enableNameShadowing */false
        );

        if (parse_result.has_value()) {
            if (failed(*parse_result))
                return mlir::failure();
            // Function body was parsed, make sure its not empty.
            if (body.empty())
                return parser.emitError(loc, "expected non-empty function body");
        }

        return mlir::success();
    }

    void printFunctionSignatureAndBody(
        Printer &printer, FuncOp op, Attribute /* funcion_type */, mlir::DictionaryAttr, Region &body
    ) {
        auto fty = op.getFunctionType();
        mlir::function_interface_impl::printFunctionSignature(
            printer, op, fty.getInputs(), /* variadic */false, fty.getResults()
        );

        mlir::function_interface_impl::printFunctionAttributes(
            printer, op, {"linkage", op.getFunctionTypeAttrName() }
        );

        if (!body.empty()) {
            printer.getStream() << " ";
            printer.printRegion( body,
                /* printEntryBlockArgs */false,
                /* printBlockTerminators */true
            );
        }
    }

} // namespace vast::ll

#define GET_OP_CLASSES
#include "vast/Dialect/LowLevel/LowLevel.cpp.inc"
