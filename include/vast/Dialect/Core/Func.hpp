// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Interfaces/FunctionImplementation.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/Linkage.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"

namespace vast::core {

    llvm::StringRef getLinkageAttrNameString();

    // This function is adapted from CIR:
    //
    // Verifies linkage types, similar to LLVM:
    // - functions don't have 'common' linkage
    // - external functions have 'external' or 'extern_weak' linkage
    template< typename FuncOp >
    logical_result verifyFuncOp(FuncOp op) {
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


    template< typename FuncOp >
    void printFunctionSignatureAndBody(
        Printer &printer, FuncOp op,
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

    template< typename DstFuncOp >
    DstFuncOp convert_function(auto src, auto &rewriter, string_ref name, core::FunctionType fty, auto body_builder) {
        mlir::SmallVector< mlir::DictionaryAttr, 8 > arg_attrs;
        mlir::SmallVector< mlir::DictionaryAttr, 8 > res_attrs;
        src.getAllArgAttrs(arg_attrs);
        src.getAllResultAttrs(res_attrs);

        auto filter_src_attrs = [&] (auto op) {
            mlir::SmallVector< mlir::NamedAttribute > result;

            for (auto attr : op->getAttrs()) {
                auto name = attr.getName();
                if (name == mlir::SymbolTable::getSymbolAttrName() ||
                    name == src.getFunctionTypeAttrName() ||
                    name == getLinkageAttrNameString() ||
                    name == src.getArgAttrsAttrName() ||
                    name == src.getResAttrsAttrName()
                ) {
                    continue;
                }

                result.push_back(attr);
            }

            return result;
        };

        auto dst = rewriter.template create< DstFuncOp >(
            src.getLoc(), name, fty, src.getLinkage(), filter_src_attrs(src), arg_attrs, res_attrs
        );

        body_builder(rewriter, dst);

        return dst;
    };

    template< typename DstFuncOp >
    DstFuncOp convert_function(auto src, auto &rewriter, string_ref name) {
        return convert_function< DstFuncOp >(src, rewriter, name, src.getFunctionType(),
            [src] (auto &rewriter, auto &dst) mutable {
                rewriter.updateRootInPlace(dst, [&] () {
                    dst.getBody().takeBody(src.getBody());
                });
            }
        );
    };

    template< typename DstFuncOp >
    DstFuncOp convert_function(auto src, auto &rewriter) {
        return convert_function< DstFuncOp >(src, rewriter, src.getName());
    }

    template< typename DstFuncOp >
    DstFuncOp convert_function_without_body(auto src, auto &rewriter, string_ref name, core::FunctionType fty) {
        return convert_function< DstFuncOp >(src, rewriter, name, fty, [] (auto &, auto &) {} /* noop */ );
    };

    template< typename DstFuncOp >
    logical_result convert_and_replace_function(auto src, auto &rewriter, string_ref name) {
        auto dst = convert_function< DstFuncOp >(src, rewriter, name);
        rewriter.replaceOp(src, dst->getOpResults());
        return mlir::success();
    }

    template< typename DstFuncOp >
    logical_result convert_and_replace_function(auto src, auto &rewriter) {
        return convert_and_replace_function< DstFuncOp >(src, rewriter, src.getName());
    }

} // namespace vast::core
