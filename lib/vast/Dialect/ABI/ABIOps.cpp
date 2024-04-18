// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/ABI/ABIDialect.hpp"
#include "vast/Dialect/ABI/ABIOps.hpp"

#include "vast/Util/Region.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Dialect.hpp"

#include "vast/Dialect/Core/Func.hpp"

namespace vast::abi
{
    ParseResult parseFunctionSignatureAndBody(
        Parser &parser, Attribute &funcion_type,
        mlir::NamedAttrList &attr_dict, Region &body
    ) {
        return core::parseFunctionSignatureAndBody( parser, funcion_type, attr_dict, body );
    }


    void printFunctionSignatureAndBody(
        Printer &printer, auto op,
        Attribute attr, mlir::DictionaryAttr dict, Region &body
    ) {
        return core::printFunctionSignatureAndBody( printer, op, attr, dict, body );
    }

    mlir::CallInterfaceCallable CallOp::getCallableForCallee()
    {
        return (*this)->getAttrOfType< mlir::SymbolRefAttr >("callee");
    }

    mlir::Operation::operand_range CallOp::getArgOperands()
    {
        return this->getOperands();
    }

    void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee)
    {
        setOperand(0, callee.get< mlir_value >());
    }

    void build_op_with_region(Builder &bld, State &st, BuilderCallback body_cb)
    {
        Builder::InsertionGuard guard(bld);
        auto body = st.addRegion();
        if (body_cb.has_value())
        {
            bld.createBlock(body);
            body_cb.value()(bld, st.location);
        }
    }

    void PrologueOp::build(Builder &bld, State &st,
                           mlir::TypeRange types, BuilderCallback body_cb)
    {
        st.addTypes(types);
        return build_op_with_region(bld, st, body_cb);
    }

    void EpilogueOp::build(Builder &bld, State &st,
                           mlir::TypeRange types, BuilderCallback body_cb)
    {
        st.addTypes(types);
        return build_op_with_region(bld, st, body_cb);
    }

    void CallArgsOp::build(Builder &bld, State &st,
                           mlir::TypeRange types, BuilderCallback body_cb)
    {
        st.addTypes(types);
        return build_op_with_region(bld, st, body_cb);
    }

    void CallRetsOp::build(Builder &bld, State &st,
                           mlir::TypeRange types, BuilderCallback body_cb)
    {
        st.addTypes(types);
        return build_op_with_region(bld, st, body_cb);
    }

    void CallExecutionOp::build(Builder &bld, State &st, llvm::StringRef callee,
                                mlir::TypeRange types, mlir::ValueRange operands,
                                BuilderCallback body_cb)
    {
        st.addTypes(types);
        build_op_with_region(bld, st, body_cb);
        st.addOperands(operands);
        st.addAttribute("callee", mlir::SymbolRefAttr::get(bld.getContext(), callee));
    }

    mlir::CallInterfaceCallable CallExecutionOp::getCallableForCallee()
    {
        return (*this)->getAttrOfType< mlir::SymbolRefAttr >("callee");
    }

    mlir::Operation::operand_range CallExecutionOp::getArgOperands()
    {
        return this->getOperands();
    }

    void CallExecutionOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee)
    {
        setOperand(0, callee.get< mlir_value >());
    }

    SSACFG_REGION_OP( FuncOp );

    logical_result FuncOp::verify() {
        return core::verifyFuncOp(*this);
    }

} // namespace vast::abi

#define GET_OP_CLASSES
#include "vast/Dialect/ABI/ABI.cpp.inc"
